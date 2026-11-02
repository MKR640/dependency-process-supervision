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
    datasets = sum([all_datasets[name] for name in selected_names if name in all_datasets], [])
else:
    datasets = sum(all_datasets.values(), [])

# datasets = sum((v for k, v in locals().items() if k.endswith("_datasets")), [])

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>', generate=True),
    ],
    begin="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    eos_token_id=151645
)



models = [
    dict(
        type=VLLM,
        abbr='Ours',
        path="models/sft_merge_lora",
        model_kwargs=dict(gpu_memory_utilization=0.9, max_model_len=32768),
        generation_kwargs=dict(temperature=0),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        meta_template=_meta_template,
        stop_words=['<|im_end|>', '<|endoftext|>']
    )
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000),
    runner=dict(
        type=LocalRunner, max_num_workers=8,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask)),
)
