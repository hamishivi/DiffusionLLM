from interactive import InteractiveDiffusion
from ifeval import test_instruction_following_strict, test_instruction_following_loose, load_ifeval_prompts
import collections


model_args = "flan_v2.xxl.length/args.json"
model_ckpt = "flan_v2.xxl.length/checkpoint-20000"

engine = InteractiveDiffusion(model_args, model_ckpt)
tokenizer = engine.tokenizer

input_data = load_ifeval_prompts()

def calculate_scores(os):
    """Helper function to calculate accuracy scores from outputs.
    
    Args:
        outputs (list): List of OutputExample objects
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    prompt_total = len(os)
    prompt_correct = sum(1 for o in os if o.follow_all_instructions)
    
    instruction_total = sum(len(o.instruction_id_list) for o in os)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in os)

    # Calculate per-instruction accuracies
    instruction_metrics = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    
    for output in os:
        for inst_id, followed in zip(output.instruction_id_list, output.follow_instruction_list):
            instruction_metrics[inst_id]["total"] += 1
            if followed:
                instruction_metrics[inst_id]["correct"] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total,
        "instruction_level_accuracy": instruction_correct / instruction_total,
        "per_instruction_accuracy": {
            k: v["correct"] / v["total"]
            for k, v in instruction_metrics.items()
        }
    }

outputs = []
for sample in input_data:
    prompt = sample.prompt
    input_len = len(tokenizer.encode(prompt))
    if input_len > 512:
        print(f"input length is {input_len}, skip")
        outputs.append("")
        continue
    answer = ""
    engine_o = engine.sample(prompt, f"{512 - input_len} ")
    for step in engine_o:
        answer = step
    outputs.append(answer)

loose_outputs, strict_outputs = [], []
for i, sample in enumerate(input_data):
    # Test instruction following in strict mode
    strict_result = test_instruction_following_strict(
        sample,
        outputs[i]
    )
    strict_outputs.append(strict_result)
    # Test instruction following in loose mode 
    loose_result = test_instruction_following_loose(
        sample,
        outputs[i]
    )
    loose_outputs.append(loose_result)

# Calculate metrics for both strict and loose evaluation
metrics = {}
strict_metrics = calculate_scores(strict_outputs)
for k, v in strict_metrics.items():
    metrics[f"strict_{k}"] = v
loose_metrics = calculate_scores(loose_outputs)
for k, v in loose_metrics.items():
    metrics[f"loose_{k}"] = v
print(metrics)
