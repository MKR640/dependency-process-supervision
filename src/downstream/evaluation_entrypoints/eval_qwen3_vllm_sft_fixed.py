from opencompass.models import VLLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from mmengine.config import read_base

with read_base():
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from .datasets.race.race_gen_69ee4f import race_datasets
    from .datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.math.math_gen_265cce import math_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.mbpp.mbpp_gen_830460 import mbpp_datasets
    from .datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from .datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets


all_datasets = {
    'CEval': ceval_datasets,
    'AGIEval': agieval_datasets,
    'MMLU': mmlu_datasets,
    'CMMLU': cmmlu_datasets,
    'GaokaoBench': GaokaoBench_datasets,
    'TriviaQA': triviaqa_datasets,
    'RACE': race_datasets,
    'Lambada': lambada_datasets,
    'Hellaswag': hellaswag_datasets,
    'Math': math_datasets,
    'GSM8K': gsm8k_datasets,
    'HumanEval': humaneval_datasets,
    'MBPP': mbpp_datasets,
    'BBH': bbh_datasets,
    'IFEval': ifeval_datasets,
}


selected = ""
if selected:
    selected_names = [x.strip() for x in selected.split(",")]
    datasets = sum(
        [all_datasets[name] for name in selected_names if name in all_datasets],
        [],
    )
else:
    datasets = sum(all_datasets.values(), [])


# -----------------------------------------------------------------------------
# Qwen3-4B-Base / MATH fix
# -----------------------------------------------------------------------------
# Qwen3-4B-Base is a PRETRAINED BASE model.  The MATH dataset config already
# provides the official-style 4-shot CoT completion prompt:
#
#     Problem: ...
#     Solution: ...
#     ...
#     Problem: {problem}
#     Solution:
#
# Therefore DO NOT add a ChatML meta_template such as
# <|im_start|>user / <|im_start|>assistant around these few-shot examples.
# Doing so makes the base model continue the conversation after answering the
# test problem and regenerate the few-shot examples.  math_postprocess_v2 then
# tends to extract a later demonstration answer (in this prompt, often "24")
# instead of the answer to the current problem.
#
# Keep the official MATH prompt/evaluator untouched, but give MATH a little more
# generation room.  The original dataset config uses 512 tokens.
for _dataset in math_datasets:
    _dataset['infer_cfg']['inferencer']['max_out_len'] = 1024


models = [
    dict(
        type=VLLM,
        abbr='Ours',
        path="models/Qwen3-4B-base",

        # Qwen3-4B-Base has a 32K pretraining evaluation context.  8192 here is
        # already enough for the 4-shot MATH prompt + a long completion and
        # avoids accidental truncation caused by the previous max_seq_len=2048.
        model_kwargs=dict(
            gpu_memory_utilization=0.9,
            max_model_len=32768,
        ),

        # Base-model evaluation: deterministic greedy decoding.
        # Do NOT use the Qwen3 post-trained thinking-mode sampling recipe
        # (temperature=0.6/top_p=0.95) for Qwen3-4B-Base reproduction.
        generation_kwargs=dict(
            temperature=0,
        ),

        max_out_len=2048,
        max_seq_len=8192,
        batch_size=8,
        run_cfg=dict(num_gpus=1),

        # IMPORTANT: no meta_template for Qwen3-4B-Base.
        # Let OpenCompass feed the few-shot completion prompt directly.

        # Prevent a completion-style base model from starting another benchmark
        # example after it has finished the current answer.  <|endoftext|> is
        # also kept as the normal text EOS marker.
        stop_words=[
            '\nProblem:',
            '\nQuestion:',
            '<|endoftext|>',
        ],
    )
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask),
    ),
)
