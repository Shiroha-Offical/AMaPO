# Adaptive Margin-attached Preference Optimization (AMaPO)

This repository contains the code for our paper. We propose a simpler and more effective preference optimization algorithm than DPO (Direct Preference Optimization) and SimPO (Simple Preference Optimization) without external tuning hyperparameters. AMaPO outperforms DPO and its latest variants across **Preference Ranking** and **Alignment Downstream Task** benchmarks under various settings. We will release our full training code after acceptance

## Environment
We provide an environment file in `environment.yml` including the python package versions we used in our experiments. For optimal reproducibility, we recommend using the same package versions. However, please note that results may still vary due to differences in hardware configurations and CUDA versions, etc.

## Hyperparameter tuning
Hyperparameter tuning is crucial for AMaPO (and other preference optimization algorithms in general). The two main hyperparameters of AMaPO to focus on are `learning_rate`, `beta` (we recommend keeping the total batch size fixed at 128).
- `learning_rate`: It is the most critical hyperparameter for preference optimization. A large learning rate (e.g., 1e-5) can significantly degrade performance, causing the model to produce incoherent sentences or completely repetitive responses. We recommend grid searching over 4e-7, 6e-7, 8e-7, and 1e-6, if resources allow.
- `beta`: Beta controls the reward scaling between winning and losing responses. AMaPO requires a close `beta` to SimPO. We recommend a default beta of `2.0` or `3.0`.

We used the following hyperparameters for training the reported results as shown in `training_configs`.
| Setting           | β   | Learning rate |
|-------------------|-----|----------------|
| Mistral-Base      | 2.0 | 4e-7           |
| Mistral-Instruct  | 2.0 | 6e-7           |
| Llama3-Base       | 2.5 | 6e-7           |
| Llama3-Instruct   | 3.0 | 1e-6           |

## Training Scripts 
We provide four training config files for the four training setups reported in our paper. The training config is set for 4xA100 GPUs. You may need to adjust `num_processes` and `per_device_train_batch_size` based on your computation environment. 

**Note that the amapo_trainer.py will be released after acceptance**

* Mistral-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_AMaPO.py training_configs/mistral-7b-base-AMaPO.yaml
```
* Mistral-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_AMaPO.py training_configs/mistral-7b-instruct-AMaPO.yaml
```
* Llama3-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_AMaPO.py training_configs/llama-3-8b-base-AMaPO.yaml
```
* Llama3-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_AMaPO.py training_configs/llama-3-8b-instruct-AMaPO.yaml
```

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, MT-Bench, RM-Bench, open_llm_leaderboard and RM-robustness (more details can be found under `\eval\README.md`):

## Acknowledgement
We build our project based on following repositories
* [SimPO](https://github.com/princeton-nlp/SimPO)
* [The Alignment Handbook](https://github.com/huggingface/alignment-handbook)