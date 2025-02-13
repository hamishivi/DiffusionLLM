from interactive import InteractiveDiffusion
import argparse
from datasets import load_dataset
import evaluate
from squad_eval_1 import evaluate

model_args = "flan_v2.xxl.length/args.json"
model_ckpt = "flan_v2.xxl.length/checkpoint-20000"

engine = InteractiveDiffusion(model_args, model_ckpt)
tokenizer = engine.tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hamishivi/squad_eval_diffulm")
args = parser.parse_args()

ds = load_dataset(args.dataset, split="train").shuffle(seed=42).select(range(512))

outputs = []
for sample in ds:
    input_len = len(tokenizer.encode(sample["inputs"]))
    # based on xlm, so 512 is max length
    assert input_len < 512
    answer = ""
    print(sample["inputs"])
    for step in engine.sample(sample["inputs"], f"{512 - input_len} "):
        answer = step
    outputs.append(answer)
    print(answer)


references = [{"id": sample["id"], "answers": sample["answers"]} for sample in ds]
outputs = [{"id": ds[i]["id"], "prediction_text": output} for i, output in enumerate(outputs)]
results = evaluate(references=references, predictions=outputs)
print(results)
