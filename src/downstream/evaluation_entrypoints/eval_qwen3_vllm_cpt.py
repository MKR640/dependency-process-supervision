from opencompass.models import VLLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.ceval.ceval_ppl_578f8d import ceval_datasets
    from opencompass.configs.datasets.agieval.agieval_gen_64afd3 import agieval_datasets
    from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from opencompass.configs.datasets.race.race_ppl_abed12 import race_datasets
    from opencompass.configs.datasets.lambada.lambada_gen_217e11 import lambada_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_zeroshot_datasets
    from opencompass.configs.datasets.math.math_gen_265cce import math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from opencompass.configs.datasets.mbpp.mbpp_gen_830460 import mbpp_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets
    from opencompass.configs.datasets.SeedBench.seedbench_gen_5d5ea1 import seedbench_datasets
    from opencompass.configs.datasets.agxqa.agxqa_gen import agxqa_datasets
    from opencompass.configs.datasets.bioprobench.bioprobench_mixed import BioProbench_datasets
    from opencompass.configs.datasets.ProteinLMBench.ProteinLMBench_gen_a67965 import proteinlmbench_datasets
    from opencompass.configs.datasets.ChemBench4K.ChemBench_ppl import chembench4k_datasets
    from opencompass.configs.datasets.CS_Bench.CS_Bench_mixed import CS_Bench_datasets
    from opencompass.configs.datasets.AECBench.AECBenchData_gen_16afed import AECBenchData_datasets
    from opencompass.configs.datasets.enmb.enmb_gen import enmb_datasets
    from opencompass.configs.datasets.ClimaQA.ClimaQA_llm_judge_gen import climaqa_datasets
    from opencompass.configs.datasets.EarthSE.EarthSE_ppl import earthse_mcq_dataset
    from opencompass.configs.datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import aime2025_datasets
    from opencompass.configs.datasets.mascqa.mascqa_mixed import MaScQA_datasets
    from opencompass.configs.datasets.matbench.matbench_ppl import matbench_datasets
    from opencompass.configs.datasets.MedQA.MedQA_ppl import MedQA_datasets
    from opencompass.configs.datasets.PubMedQA.PubMedQA_ppl import PubMedQA_datasets
    from opencompass.configs.datasets.medmcqa.medmcqa_ppl import medmcqa_datasets
    from opencompass.configs.datasets.ScienceQA.ScienceQA_ppl import ScienceQA_datasets
    from opencompass.configs.datasets.gpqa.gpqa_few_shot_ppl_4b5a83 import gpqa_datasets
    from opencompass.configs.datasets.SciEval.SciEval_5shot_ppl import scieval_datasets
    from opencompass.configs.datasets.scibench.scibench_gen_2b21f3 import scibench_datasets
    from opencompass.configs.datasets.SciKnowEval.SciKnowEval_ppl import sciknoweval_datasets
    from opencompass.configs.datasets.sciq.sciq_ppl import sciq_datasets
    from opencompass.configs.datasets.astro_qa.astro_qa_260206 import astro_qa_datasets
    from opencompass.configs.datasets.astro_mcq.astro_mcq_260206 import astro_mcq_datasets
    from opencompass.configs.datasets.astro_bench_test.astro_bench_test_ppl import astro_bench_datasets
    from opencompass.configs.datasets.eese.eese_llm_judge_gen import eese_datasets   #数据集有EESE、EESE-V2两个版本，仅评测EESE版本
    from opencompass.configs.datasets.prost.prost_ppl import prost_datasets
    from opencompass.configs.datasets.PHYBench.phybench_gen import phybench_datasets
    from opencompass.configs.datasets.PHYSICS.PHYSICS_llm_judge_gen_260204 import physics_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_ppl_a6e128 import hellaswag_tenshot_datasets
    from opencompass.configs.datasets.coqa.coqa_810ec6 import coqa_datasets
    from opencompass.configs.datasets.bigbench_qa_wikidata.bigbench_qa_wikidata_gen_217e11 import bigbench_qa_wikidata_datasets
    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_ppl_a450bd import ARC_c_datasets
    from opencompass.configs.datasets.obqa.obqa_ppl_c7c154 import obqa_datasets
    from opencompass.configs.datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    from opencompass.configs.datasets.ESGenius.ESGenius_ppl import esgenius_ppl_datasets
    from opencompass.configs.datasets.HydroLLM.HydroLLM_mcq import hydro_mcq_ppl_datasets
    from opencompass.configs.datasets.HydroLLM.HydroLLM_tf  import hydro_tf_ppl_datasets
    from opencompass.configs.datasets.HydroLLM.HydroLLM_fitb      import hydro_fitb_datasets
    from opencompass.configs.datasets.HydroLLM.HydroLLM_openended import hydro_openended_datasets
    from opencompass.configs.datasets.SocialIQA.SocialIQA_ppl import socialiqa_ppl_datasets
    from opencompass.configs.datasets.literaryqa.literaryqa_gen import literaryqa_datasets
    from opencompass.configs.datasets.AtmosSciBench.AtmosSciBench_mcq_ppl import atmosci_bench_mcq_datasets
    from opencompass.configs.datasets.AtmosSciBench.AtmosSciBench_oeq_gen import atmosci_bench_oeq_datasets
    from opencompass.configs.datasets.oneatmos.oneatmos_tf_ppl import oneatmos_tf_ppl_datasets
    from opencompass.configs.datasets.oneatmos.oneatmos_open_gen import oneatmos_open_datasets
    from opencompass.configs.datasets.mentat.mentat_ppl import mentat_ppl_datasets
    from opencompass.configs.datasets.CivilBench.CivilBench_sc_ppl     import civilbench_sc_ppl_datasets
    from opencompass.configs.datasets.CivilBench.CivilBench_mc_gen   import civilbench_mc_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_std_ppl           import fsiq_std_ppl_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_std10_ppl         import fsiq_std10_ppl_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_std_mc_gen        import fsiq_std_mc_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_pert_simple_ppl   import fsiq_pert_simple_ppl_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_pert_complex_ppl  import fsiq_pert_complex_ppl_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_pert10_simple_ppl import fsiq_pert10_simple_ppl_datasets
    from opencompass.configs.datasets.FailureSensorIQ.FailSensorIQ_pert10_complex_ppl import fsiq_pert10_complex_ppl_datasets
    from opencompass.configs.datasets.ORQA.ORQA_ppl import orqa_ppl_datasets
    from opencompass.configs.datasets.AgentDrive.AgentDrive_ppl import agentdrive_ppl_datasets
    from opencompass.configs.datasets.FormationEval.FormationEval_v01_ppl import formationeval_v01_ppl_datasets
    from opencompass.configs.datasets.FormationEval.FormationEval_spe_ppl import formationeval_spe_ppl_datasets
    from opencompass.configs.datasets.FormationEval.FormationEval_diskos_gen import formationeval_diskos_gen_datasets
    from opencompass.configs.datasets.ElecBench.ElecBench_oeq_gen   import elecbench_oeq_datasets
    from opencompass.configs.datasets.ElecBench.ElecBench_mcq_ppl   import elecbench_mcq_ppl_datasets
    from opencompass.configs.datasets.BizCompass.BizCompass_mixed import bizcompass_datasets
    from opencompass.configs.datasets.CulturalBench.CulturalBench_ppl import culturalbench_ppl_datasets
    from opencompass.configs.datasets.DFIR_Metric.DFIR_Metric_ppl import dfir_metric_ppl_datasets
    from opencompass.configs.datasets.EconLogicQA.EconLogicQA_gen_few_shot import econlogicqa_gen_few_shot_datasets
    from opencompass.configs.datasets.LEXam.LEXam_ppl import lexam_ppl_datasets
    from opencompass.configs.datasets.UNBench.UNBench_ppl import unbench_ppl_datasets
    from opencompass.configs.datasets.xiezhi.xiezhi_ppl_raw_aggregation import xiezhi_datasets
    from opencompass.configs.datasets.MSQA.MSQA_ppl import msqa_ppl_datasets
    from opencompass.configs.datasets.ALUE.ALUE_ppl import alue_ppl_datasets
    from opencompass.configs.datasets.ScholarChemQA.ScholarChemQA_ppl import scholarchemqa_ppl_datasets
    from opencompass.configs.datasets.FoodieQA.FoodieQA_ppl import foodieqa_ppl_datasets
    from opencompass.configs.datasets.QCBench.QCBench_gen_few_shot import qcbench_few_shot_gen_datasets
    from opencompass.configs.datasets.UGPhysics.UGPhysics_ppl import ugphysics_ppl_datasets
    from opencompass.configs.datasets.UGPhysics.UGPhysics_gen_few_shot import ugphysics_numeric_gen_datasets

all_datasets = {
    'CEval': ceval_datasets,
    'AGIEval': agieval_datasets,
    'MMLU': mmlu_datasets,
    'CMMLU': cmmlu_datasets,
    'GaokaoBench': GaokaoBench_datasets,
    'TriviaQA': triviaqa_datasets,
    'RACE': race_datasets,
    'Lambada': lambada_datasets,
    'HellaSwag_zeroshot': hellaswag_zeroshot_datasets,
    'Math': math_datasets,
    'GSM8K': gsm8k_datasets,
    'HumanEval': humaneval_datasets,
    'MBPP': mbpp_datasets,
    'BBH': bbh_datasets,
    'IFEval': ifeval_datasets,
    'SeedBench': seedbench_datasets,
    'AgXQA': agxqa_datasets,
    'BioproBench': BioProbench_datasets,
    'ProteinLMBench': proteinlmbench_datasets,
    'ChemBench4K': chembench4k_datasets,
    'CS_Bench': CS_Bench_datasets,
    'AECBench': AECBenchData_datasets,
    'ENMB': enmb_datasets,
    'ClimaQA': climaqa_datasets,
    'EarthSE': earthse_mcq_dataset,
    'AIME2025': aime2025_datasets,
    'MaScQA': MaScQA_datasets,
    'MatBench': matbench_datasets,
    'MedQA': MedQA_datasets,
    'PubMedQA': PubMedQA_datasets,
    'MedMcQA': medmcqa_datasets,
    'ScienceQA': ScienceQA_datasets,
    'GPQA': gpqa_datasets,
    'SciEval': scieval_datasets,
    'SciBench': scibench_datasets,
    'SciKnowEval': sciknoweval_datasets,
    'SciQ': sciq_datasets,
    'Astro_QA': astro_qa_datasets,
    'Astro_MCQ': astro_mcq_datasets,
    'Astrobench': astro_bench_datasets,
    'EESE': eese_datasets,
    'Prost': prost_datasets,
    'PhyBench': phybench_datasets,
    'Physics': physics_datasets,
    'HellaSwag_tenshot': hellaswag_tenshot_datasets,
    'BigBench_QA_wikiData': bigbench_qa_wikidata_datasets,
    'CoQA': coqa_datasets,
    'ARC_Easy': ARC_e_datasets,
    'ARC_Challenge': ARC_c_datasets,
    'OpenBookQA':obqa_datasets,
    'PIQA':piqa_datasets,
    'ESGenius':esgenius_ppl_datasets,
    'HydroLLM': (hydro_mcq_ppl_datasets + hydro_tf_ppl_datasets +
                 hydro_fitb_datasets + hydro_openended_datasets),
    'SocialIQA': socialiqa_ppl_datasets,
    'LiteraryQA': literaryqa_datasets,
    'AtmosSciBench': (atmosci_bench_mcq_datasets +
                      atmosci_bench_oeq_datasets),
    'OneAtmos': oneatmos_open_datasets + oneatmos_tf_ppl_datasets,
    'MENTAT': mentat_ppl_datasets,
    'CivilBench': civilbench_sc_ppl_datasets + civilbench_mc_datasets,
    'FailSensorIQ': (fsiq_std_ppl_datasets + fsiq_std10_ppl_datasets +
                     fsiq_std_mc_datasets + fsiq_pert_simple_ppl_datasets +
                     fsiq_pert_complex_ppl_datasets +
                     fsiq_pert10_simple_ppl_datasets +
                     fsiq_pert10_complex_ppl_datasets),
    'ORQA': orqa_ppl_datasets,
    'AgentDrive': agentdrive_ppl_datasets,
    'FormationEval': (formationeval_v01_ppl_datasets +
                      formationeval_spe_ppl_datasets +
                      formationeval_diskos_gen_datasets),
    'ElecBench': elecbench_oeq_datasets + elecbench_mcq_ppl_datasets,
    'BizCompass': bizcompass_datasets,
    'EconLogicQA': econlogicqa_gen_few_shot_datasets,
    'LEXam': lexam_ppl_datasets,
    'UNBench': unbench_ppl_datasets,
    'DFIR-Metric': dfir_metric_ppl_datasets,
    'CulturalBench': culturalbench_ppl_datasets,
    'Xiezhi': xiezhi_datasets,
    'MSQA': msqa_ppl_datasets,
    'ALUE': alue_ppl_datasets,
    'ScholarChemQA': scholarchemqa_ppl_datasets,
    'FoodieQA': foodieqa_ppl_datasets,
    'QCBench': qcbench_few_shot_gen_datasets,
    'UGPhysics': ugphysics_ppl_datasets + ugphysics_numeric_gen_datasets,
}


selected = ""
if selected:
    selected_names = [x.strip() for x in selected.split(",")]
    datasets = sum([all_datasets[name] for name in selected_names if name in all_datasets], [])
else:
    datasets = sum(all_datasets.values(), [])


models = [
    dict(
        type=VLLM,
        abbr='Ours',
        path="models/Qwen3-0.6B-Base",
        model_kwargs=dict(gpu_memory_utilization=0.75, max_model_len=32768),
        generation_kwargs=dict(temperature=0),
        max_out_len=2048, #max_tokens
        max_seq_len=2048, #mode=mid，prompt截断，保留prompt两端信息；默认mode=none，max_seq_len无用，prompt不截断
        batch_size=8, 
        run_cfg=dict(num_gpus=1),
        # meta_template=_meta_template,
        # stop_words=['<|im_end|>', '<|endoftext|>']
    )
]



infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=50000),
    runner=dict(
        type=LocalRunner, max_num_workers=8,
        max_workers_per_gpu=1,
        task=dict(type=OpenICLInferTask)),
)
