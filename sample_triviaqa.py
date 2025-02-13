from interactive import InteractiveDiffusion
import re
from datasets import load_dataset
import string


model_args = "flan_v2.xxl.length/args.json"
model_ckpt = "flan_v2.xxl.length/checkpoint-20000"

engine = InteractiveDiffusion(model_args, model_ckpt)
tokenizer = engine.tokenizer

ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
ds = ds.shuffle(seed=42).select(range(2000))

triviaqa_shots = [
    "Which American-born Sinclair won the Nobel Prize for Literature in 1930?\n\n(Harry) Sinclair Lewis",
    "Where in England was Dame Judi Dench born?\n\nYork, England",
]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# create new dataset with inputs and targets
new_data = []
for sample in ds:
    new_data.append({
        "inputs": "\n".join(triviaqa_shots) + "\n\n" + sample["question"],
        "targets": sample["answer"]["aliases"][0]
    })

outputs = []
for sample in new_data:
    input_len = len(tokenizer.encode(sample["inputs"]))
    # based on xlm, so 512 is max length
    assert input_len < 512
    answer = ""
    print(sample["inputs"])
    engine_o = engine.sample(sample["inputs"], f"{512 - input_len} ")
    for step in engine_o:
        answer = step
    outputs.append(answer)
    print(answer)

# just use diffullama em style
correct = 0
for answer, samples in zip(outputs, new_data):
    for answer_opt in samples["answer"]["aliases"]:
        if normalize_answer(answer) == normalize_answer(answer_opt):
            correct += 1
            break

print(f"Exact match : {correct / len(outputs)}")