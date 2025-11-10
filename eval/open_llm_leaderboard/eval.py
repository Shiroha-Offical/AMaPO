import os
import lm_eval
import json
import torch
import argparse

save_eval_dir = "your/results/dir"
def save_results(save_path, results):
    with open(save_path, 'w') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def eval_tasks(model, task):
    llm = lm_eval.models.vllm_causallms.VLLM(
        pretrained=model,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        # data_parallel_size=1,
        gpu_memory_utilization=0.4,
    )
    # Evaluation on benchmarks
    task_manager = lm_eval.tasks.TaskManager()

    results = lm_eval.simple_evaluate(
        model=llm,
        tasks=task,
        batch_size='auto',
        task_manager=task_manager,
    )

    save_dir = os.path.join(save_eval_dir, model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{task}.json")
    save_results(save_path, results)

def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model_name")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # tasks in open_llm_leaderboard v1 and v2
    tasks = ["truthfulqa_mc2", "arc_challenge", "gsm8k", "winogrande", "leaderboard"]
    # path/to/your/model
    model = args.model
    for task in tasks:
        eval_tasks(model, task)