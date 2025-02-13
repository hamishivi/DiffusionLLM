from interactive import InteractiveDiffusion
import argparse
from datasets import load_dataset
import evaluate

model_args = "flan_v2.xxl.length/args.json"
model_ckpt = "flan_v2.xxl.length/checkpoint-20000"

engine = InteractiveDiffusion(model_args, model_ckpt)
tokenizer = engine.tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hamishivi/gsm8k_eval_diffulm")
args = parser.parse_args()

ds = load_dataset(args.dataset, split="train")

outputs = []
for sample in ds:
    input_len = len(tokenizer.encode(sample["inputs"]))
    # based on xlm, so 512 is max length
    assert input_len < 512
    answer = ""
    for step in engine.sample(sample["inputs"], f"{512 - input_len} "):
        answer = step
    outputs.append(answer)

predictions = []
for output in outputs:
    # replace numbers like `x,xxx` with `xxxx`
    output = re.sub(r"(\d),(\d)", r"\1\2", output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        predictions.append(numbers[-1])
    else:
        predictions.append(output)

targets = [example["answer"] for example in ds]

exact_match = evaluate.load("exact_match")
em_score = exact_match.compute(
    predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True
)["exact_match"]
print(f"Exact match : {em_score}")