import argparse
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset


def verify_saved_model(output_dir: str) -> None:
    """Verify that the final model files are complete before reporting success."""
    output_path = Path(output_dir)

    if not output_path.is_dir():
        raise RuntimeError(f"Output directory does not exist: {output_path}")

    # Sharded safetensors checkpoint.
    safetensors_index = output_path / "model.safetensors.index.json"
    if safetensors_index.is_file():
        with safetensors_index.open("r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid weight map in {safetensors_index}")

        shard_names = sorted(set(weight_map.values()))
        missing = [name for name in shard_names if not (output_path / name).is_file()]
        empty = [name for name in shard_names if (output_path / name).is_file() and (output_path / name).stat().st_size == 0]

        if missing or empty:
            raise RuntimeError(
                f"Incomplete sharded model save. Missing shards: {missing}; empty shards: {empty}"
            )
        return

    # Sharded PyTorch checkpoint.
    pytorch_index = output_path / "pytorch_model.bin.index.json"
    if pytorch_index.is_file():
        with pytorch_index.open("r", encoding="utf-8") as f:
            index = json.load(f)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid weight map in {pytorch_index}")

        shard_names = sorted(set(weight_map.values()))
        missing = [name for name in shard_names if not (output_path / name).is_file()]
        empty = [name for name in shard_names if (output_path / name).is_file() and (output_path / name).stat().st_size == 0]

        if missing or empty:
            raise RuntimeError(
                f"Incomplete sharded model save. Missing shards: {missing}; empty shards: {empty}"
            )
        return

    # Non-sharded checkpoint.
    single_file_candidates = (
        output_path / "model.safetensors",
        output_path / "pytorch_model.bin",
        output_path / "adapter_model.safetensors",
        output_path / "adapter_model.bin",
    )
    if any(path.is_file() and path.stat().st_size > 0 for path in single_file_candidates):
        return

    raise RuntimeError(
        f"No complete model weights or weight index were found in {output_path}"
    )


def cleanup_distributed() -> None:
    """Explicitly destroy the distributed process group during process teardown."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as exc:
        # Do not hide the original training/saving exception because cleanup failed.
        print(f"Warning: failed to destroy the distributed process group cleanly: {exc}", flush=True)


def main(script_args, training_args, model_args, dataset_args):
    from accelerate.logging import get_logger
    from datasets import load_dataset

    # get_quantization_config is intentionally omitted for compatibility with
    # the TRL version used in this environment.
    from trl import SFTTrainer, get_dataset, get_peft_config

    logger = get_logger(__name__)
    trainer = None

    try:
        # 1. Configure model initialization.
        training_args.model_init_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            dtype=model_args.dtype,
        )

        # 2. Load the dataset.
        logger.info("正在分别从不同路径加载训练集和测试集...")

        # # A. 加载训练集 (jsonl 文件)
        # train_dataset = load_dataset(
        #     "json",
        #     data_files="data/gsm8k/train_cleaned.jsonl",
        #     split="train",
        # )

        # A. 从命令行参数加载训练集
        if not script_args.train_data_path:
            raise ValueError(
                "未指定训练集路径，请使用 "
                "--train_data_path /path/to/train.jsonl"
            )

        train_data_path = Path(script_args.train_data_path).expanduser().resolve()

        if not train_data_path.is_file():
            raise FileNotFoundError(
                f"训练数据文件不存在：{train_data_path}"
            )

        logger.info(f"加载训练集：{train_data_path}")

        train_dataset = load_dataset(
            "json",
            data_files={
                script_args.dataset_train_split: str(train_data_path)
            },
            split=script_args.dataset_train_split,
        )

        # B. 从命令行参数加载测试集
        test_dataset = None
        if training_args.eval_strategy != "no":
            if not script_args.eval_data_path:
                raise ValueError(
                    "已启用评测，但未指定测试集路径，请使用 "
                    "--eval_data_path /path/to/test.jsonl"
                )

            eval_data_path = Path(script_args.eval_data_path).expanduser().resolve()

            if not eval_data_path.is_file():
                raise FileNotFoundError(
                    f"测试数据文件不存在：{eval_data_path}"
                )

            logger.info(f"加载测试集：{eval_data_path}")

            test_dataset = load_dataset(
                "json",
                data_files={
                    script_args.dataset_test_split: str(eval_data_path)
                },
                split=script_args.dataset_test_split,
            )

        # C. 合并为统一的 DatasetDict
        dataset_splits = {
            script_args.dataset_train_split: train_dataset,
        }
        if test_dataset is not None:
            dataset_splits[script_args.dataset_test_split] = test_dataset

        dataset = DatasetDict(dataset_splits)
            
        def format_dataset(example):
            """
            将原始 JSONL 转成 TRL 的 conversational prompt/completion 格式。

            优先支持本实验数据格式：
                {"prompt": "...", "completion": "..."}

            同时向后兼容旧的 GSM8K 风格：
                {"question": "...", "answer": "..."}

            注意：这里不再添加要求 `#### <number>` 的 system prompt，
            以免改变当前数学 SFT 数据本身的答案格式。
            """
            if "prompt" in example and "completion" in example:
                prompt_text = example["prompt"]
                completion_text = example["completion"]
            elif "question" in example and "answer" in example:
                prompt_text = example["question"]
                completion_text = example["answer"]
            else:
                raise ValueError(
                    "数据必须包含 prompt/completion 或 question/answer 字段；"
                    f"当前字段为: {list(example.keys())}"
                )

            if not isinstance(prompt_text, str) or not prompt_text.strip():
                raise ValueError("prompt/question 必须是非空字符串")
            if not isinstance(completion_text, str) or not completion_text.strip():
                raise ValueError("completion/answer 必须是非空字符串")

            return {
                "prompt": [
                    {"role": "user", "content": prompt_text.strip()}
                ],
                "completion": [
                    {"role": "assistant", "content": completion_text.strip()}
                ],
            }

        logger.info("Applying custom prompt/completion formatting to the dataset...")
        if isinstance(dataset, dict):
            for split in dataset:
                dataset[split] = dataset[split].map(
                    format_dataset,
                    remove_columns=dataset[split].column_names,
                    desc=f"Formatting {split} split"
                )
        else:
            dataset = dataset.map(
                format_dataset,
                remove_columns=dataset.column_names,
                desc="Formatting dataset"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=True
        )
        
        # 将 <|im_end|> 设置为真实的 EOS Token
        if "<|im_end|>" in tokenizer.get_vocab():
            tokenizer.eos_token = "<|im_end|>"
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if training_args.dataset_kwargs is None:
            training_args.dataset_kwargs = {}
        training_args.dataset_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

        trainer = SFTTrainer(
            model=model_args.model_name_or_path,
            processing_class=tokenizer, # <-- 显式传入修正后的 tokenizer
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=(
                dataset[script_args.dataset_test_split]
                if training_args.eval_strategy != "no"
                else None
            ),
            peft_config=get_peft_config(model_args),
        )

        # 获取 TRL 内部 _prepare_dataset 挂载并处理好的训练集第一条数据
        if trainer.accelerator.is_main_process:
            # 从 trainer 中获取处理好的 tokenizer
            tokenizer = trainer.processing_class
            # 如果是 ProcessorMixin (如 VLM)，取其内部的 tokenizer
            if hasattr(tokenizer, "tokenizer"):
                tokenizer = tokenizer.tokenizer

            sample = trainer.train_dataset[0]

            print("\n" + "="*20 + " [TRL 真实处理结果检查] " + "="*20, flush=True)
            print("1. input_ids 长度:", len(sample["input_ids"]), flush=True)
            print("2. labels 长度:   ", len(sample["labels"]), flush=True)

            # 解码完整输入
            decoded = tokenizer.decode(sample["input_ids"])
            print("3. 文本末尾 80 个字符:", repr(decoded[-80:]), flush=True)

            # 查看最后 10 个 Token ID 以及它们对应的 Labels
            print("4. 结尾 10 个 input_ids:", sample["input_ids"][-10:], flush=True)
            print("5. 结尾 10 个 labels:   ", sample["labels"][-10:], flush=True)
            print("6. Tokenizer EOS ID:   ", tokenizer.eos_token_id, flush=True)
            print("="*60 + "\n", flush=True)

        # 4. Train.
        logger.info("Starting SFT training...")
        train_result = trainer.train()

        # Make sure every rank has left the training loop before final saving.
        trainer.accelerator.wait_for_everyone()

        # Save final train metrics/state in addition to any checkpoint-* directory.
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # 5. Save the final model. All ranks call save_model(), while Trainer
        # performs filesystem writes only on the process allowed by should_save.
        logger.info("Training finished; saving the final model...")
        trainer.save_model(training_args.output_dir)

        # Rank 0 may still be writing large shards to NAS. Other ranks must not
        # exit and close MCCL connections until that write has completed.
        trainer.accelerator.wait_for_everyone()

        if trainer.accelerator.is_main_process:
            # Explicitly save tokenizer/processor files for robustness across
            # different Transformers/TRL versions.
            processing_class = getattr(trainer, "processing_class", None)
            if processing_class is not None:
                processing_class.save_pretrained(training_args.output_dir)

            verify_saved_model(training_args.output_dir)

            # This marker is created only after model verification succeeds.
            success_marker = Path(training_args.output_dir) / "_SUCCESS"
            success_marker.write_text("training_and_final_save_completed\n", encoding="utf-8")

        # Do not let non-main ranks tear down MCCL before rank 0 finishes
        # tokenizer saving, verification, and marker creation.
        trainer.accelerator.wait_for_everyone()

        trainer.accelerator.print(
            f"Training and final model save completed successfully: {training_args.output_dir}"
        )

        # 6. Push to Hub only from the main process.
        if training_args.push_to_hub:
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                logger.warning("Push to Hub is enabled. Proceeding with network connection...")
                try:
                    trainer.push_to_hub(dataset_name=script_args.dataset_name)
                    trainer.accelerator.print(
                        f"Model pushed to the Hub: https://huggingface.co/{trainer.hub_model_id}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Failed to push to Hub (this can be normal for an offline cluster): {exc}"
                    )
            trainer.accelerator.wait_for_everyone()

    finally:
        # On the successful path all ranks have synchronized immediately above.
        # On an exception, destroy_process_group avoids leaving MCCL resources alive.
        cleanup_distributed()


# def make_parser(
#     subparsers: argparse._SubParsersAction | None = None,
#     prog: str | None = None,
# ):
#     from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser

#     dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
#     if subparsers is not None:
#         parser = subparsers.add_parser(
#             "sft",
#             help="Run the SFT training script",
#             dataclass_types=dataclass_types,
#         )
#     else:
#         parser = TrlParser(dataclass_types, prog=prog)
#     return parser

def make_parser(
    subparsers: argparse._SubParsersAction | None = None,
    prog: str | None = None,
):
    from dataclasses import dataclass, field

    from trl import (DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser,)

    @dataclass
    class CustomScriptArguments(ScriptArguments):
        train_data_path: str | None = field(
            default=None,
            metadata={
                "help": "训练数据 JSON/JSONL 文件路径"
            },
        )
        eval_data_path: str | None = field(
            default=None,
            metadata={
                "help": "测试/验证数据 JSON/JSONL 文件路径"
            },
        )

    dataclass_types = (
        CustomScriptArguments,
        SFTConfig,
        ModelConfig,
        DatasetMixtureConfig,
    )

    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft",
            help="Run the SFT training script",
            dataclass_types=dataclass_types,
        )
    else:
        parser = TrlParser(dataclass_types, prog=prog)

    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )
    main(script_args, training_args, model_args, dataset_args)