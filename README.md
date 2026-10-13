# Medical LLM

This repository gives the code to specialize a general-purpose Large Language Model (LLM), Qwen2.5-7B-Instruct, to the medical domain through Fine-Tuning (FT) on medical and pharmacological datasets. FT uses LoRA, and two different strategies: single-phase and sequential phased training, where training is split across progressively specific knowledge domains. The project runs on High Performance Computing (HPC) systems.

Two pharmacological datasets, Medicines and DDIs, are curated from public sources and used both for FT and for evaluation. Evaluation combines multiple-choice question (MCQ) datasets, and a pairwise LLM-as-a-Judge comparison. A chat interface is included for interacting with the fine-tuned model.

## Directory Structure

- **Data**: dataset preparation notebooks.
  - `MedicalQA/`: MedQA, MedMCQA, PubMedQA: test sets downloaded from their original sources. MedQA training data comes from the PMC-LLaMA instruction dataset (https://huggingface.co/datasets/axiong/pmc_llama_instructions).
  - `Drugs/`: Medicines and DDIs, built for this project: source data downloaded from Kaggle before processing ([Medicines](https://www.kaggle.com/code/sasakitetsuya/eda-11-000-medicines), [DDIs](https://www.kaggle.com/datasets/shayanhusain/200-ddi-datasetseveritymechanisms-and-alternative)). Enriching Medicines with mechanisms of action requires a DrugBank database, requested directly from DrugBank.
  - `Mixed_Dataset/`: merges all sources into a single dataset.
- **baselines_evaluation**: zero-/few-shot MCQ evaluation of LLMs, used for baseline model selection.
- **fine_tuning**: LoRA FT (single-phase and sequential, 3 phases: MedQA → Medicines → DDIs) and its evaluation scripts.
- **medical_llms_comparation**: few-shot MCQ evaluation of existing domain-specific medical LLMs.
- **llm_as_a_judge**: pairwise open-ended evaluation pipeline.
- **chat_ui**: Gradio chat interface for the fine-tuned model.

## Results of few-shot MCQ Evaluation

A hyperparameter search over LoRA rank and learning rate, with few-shot accuracy used as the primary criterion for adapter selection, result in the following selected adapters: single-phase (r=16, lr=1.1247e-4); sequential phase 1: MedQA (r=8, lr=6.3246e-4); phase 2: Medicines (r=32, lr=3.5566e-5). Phase 3 (DDIs) did not improve on phase 2 and was not adopted. After FT, zero-shot MCQ evaluations showed a high unknown-answer rate, as the models tended to generate free-text before the final answer, making it harder to parse. The 5-shot prompting regime mitigates this formatting issue by anchoring the model to the expected output format through the provided examples.

5-shot accuracy (%), baseline is Qwen2.5-7B-Instruct without FT:

| Model | MedQA | MedMCQA | PubMedQA | Medicines | DDIs | Average |
|---|---|---|---|---|---|---|
| Baseline | 58.92 | 57.17 | 64.30 | 80.57 | 47.22 | 61.64 |
| Single-phase model | **61.43** | **59.01** | **71.50** | 83.91 | **52.78** | **65.73** |
| Sequential-phase model | 60.33 | 58.40 | 70.00 | **83.99** | 50.00 | 64.54 |

*MedMCQA is restricted only to the following subjects:  Gynaecology & Obstetrics, Surgery, Medicine, Pediatrics, Pharmacology, Radiology, Orthopaedics, and Skin.

## Setup

Three environments are used, matching the SLURM scripts' `source .../bin/activate` lines:
- `requirements-x86.txt` 
- `requirements-judge.txt` 
- `requirements-chatui.txt`

## Usage

Download the LLMs and build the datasets first: run the notebooks per source (MedQA, MedMCQA, PubMedQA, Medicines, DDIs), then `Data/Mixed_Dataset/mix_data.ipynb` to merge them.

```bash
# Evaluate the baseline Qwen2.5-7B-Instruct
sbatch baselines_evaluation/scripts/Qwen2.5-7B-Instruct/run_eval_fewshot_ddis.sh
sbatch baselines_evaluation/scripts/Qwen2.5-7B-Instruct/calculate_metrics_fewshot.sh

# FT with LoRA: single-phase
sbatch fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/scripts/Mixed_Baseline/train/r16/run_mixed_1e4.sh
sbatch fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/scripts/Mixed_Baseline/evaluation/mcq/r16/run_fewshot_lr1e4.sh
sbatch fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/scripts/Mixed_Baseline/evaluation/mcq/r16/run_extract_metrics_fewshot_lr1e4.sh

# FT with LoRA: sequential phased (phase 1, then phase 2)
sbatch fine_tuning/fine_tuning_sequential/vanilla/scripts/Qwen2.5-7B-Instruct/train/phase1_medqa/r8/run_phase1_1e4.sh
sbatch fine_tuning/fine_tuning_sequential/vanilla/scripts/Qwen2.5-7B-Instruct/train/phase2_medicines/r32/run_phase1_1e5.sh
sbatch fine_tuning/fine_tuning_sequential/vanilla/scripts/Qwen2.5-7B-Instruct/evaluation/mcq/phase2_medqa/r32/run_fewshot_lr1e5.sh
sbatch fine_tuning/fine_tuning_sequential/vanilla/scripts/Qwen2.5-7B-Instruct/evaluation/mcq/phase2_medqa/r32/run_extract_metrics_fewshot_lr1e5.sh

# Evaluation of domain-specific medical LLMs 
sbatch medical_llms_comparation/scripts/.../run_fewshot.sh
sbatch medical_llms_comparation/scripts/.../run_extract_metrics_fewshot.sh

# Generate open-ended answers (both single-phase and sequential-phase models) before judging
sbatch fine_tuning/fine_tuning_sequential/vanilla/scripts/Qwen2.5-7B-Instruct/evaluation/long_form/phase2_medicines/r32/lr_1e5/long_form_evaluation_ddis.sh
sbatch fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/scripts/Mixed_Baseline/evaluation/long_form/r16/long_form_evaluation_ddis.sh

# Open-ended evaluation with LLM-as-a-Judge (e.g: with Gemma-3-27B)
sbatch llm_as_a_judge/scripts/pairwise_criteria/Mixed_lr1.1247e-4_rank16_vs_Vanilla_P2_lr3.5566e-5_rank32/run_judge_ddis_gemma3_27b.sh
sbatch llm_as_a_judge/scripts/extract_answers/run_extract_pairwise_results.sh
sbatch llm_as_a_judge/scripts/pairwise_criteria/Mixed_lr1.1247e-4_rank16_vs_Vanilla_P2_lr3.5566e-5_rank32/run_plot.sh

# Chat with the fine-tuned single-phase model
sbatch chat_ui/run_chat_ui.sh
```

# Disclaimer

This repository was developed as part of a master's thesis. The models and code are research prototypes, intended to support, not replace, clinical judgment. They may produce inaccurate or incomplete information, including hallucinations.
