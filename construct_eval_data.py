'''
Making some eval data bits
'''
from exemplars import EXAMPLARS as GSM_EXAMPLARS
from datasets import load_dataset, Dataset


## GSM8k eval
# demonstrations = []
# for example in GSM_EXAMPLARS[:3]:
#     demonstrations.append("Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"])
# prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
# ds = load_dataset("openai/gsm8k", "main", split="test")
# new_data = []
# for sample in ds:
#     new_data.append({
#         "inputs": prompt_prefix + "Question: " + sample["question"] + "\nAnswer: ",
#         "targets": sample["answer"].split("###")[-1].strip()
#     })
# gsm8k_ds = Dataset.from_list(new_data)
# gsm8k_ds.push_to_hub("hamishivi/gsm8k_eval_diffulm")

## Squad eval
SQUAD_SHOTS = [
    "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nTo whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\n\nSaint Bernadette Soubirous",
    "Burke was born in Dublin, Ireland. His mother Mary née Nagle (c. 1702 – 1770) was a Roman Catholic who hailed from a déclassé County Cork family (and a cousin of Nano Nagle), whereas his father, a successful solicitor, Richard (died 1761), was a member of the Church of Ireland; it remains unclear whether this is the same Richard Burke who converted from Catholicism. The Burke dynasty descends from an Anglo-Norman knight surnamed de Burgh (latinised as de Burgo) who arrived in Ireland in 1185 following Henry II of England's 1171 invasion of Ireland.\n\nWhere was Burke born?\n\nDublin, Ireland",
    "The term high definition once described a series of television systems originating from August 1936; however, these systems were only high definition when compared to earlier systems that were based on mechanical systems with as few as 30 lines of resolution. The ongoing competition between companies and nations to create true \"HDTV\" spanned the entire 20th century, as each new system became more HD than the last.In the beginning of the 21st century, this race has continued with 4k, 5k and current 8K systems.\n\nThe term \"high definition\" originally described televisions systems from what year?\n\n1936"
]

ds = load_dataset("rajpurkar/squad", split="validation")
new_data = []
for sample in ds:
    prompt = sample["context"] + "\n\n" + sample["question"] + "\n\n"
    new_data.append({
        "inputs": prompt,
        "targets": sample["answers"]["text"][0]
    })
squad_ds = Dataset.from_list(new_data)
squad_ds.push_to_hub("hamishivi/squad_eval_diffulm")

# ## AlpacaEval eval
# # for this, there is no target, so we just use the prompt to get generations and will handle scoring later
# ds = load_dataset("tatsu-lab/alpaca_eval")["eval"]
# new_data = []
# for sample in ds:
#     new_data.append({
#         "inputs": sample["instruction"] + "\nResponse: ",
#         "targets": "some text"
#     })
# alpaca_ds = Dataset.from_list(new_data)
# alpaca_ds.push_to_hub("hamishivi/alpaca_eval_diffulm")
