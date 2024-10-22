import pandas as pd
import json
from datasets import load_dataset
from llama_cpp import Llama
from tqdm.auto import tqdm


d = load_dataset("facebook/anli")
dd = d['train_r2'].filter(lambda example: example["label"] == 2)

m = Llama("/home/alena.sukhanova/LLama3-8b-Instruct.gguf",
          use_mmap=False, use_mlock=True,
          n_ctx=8192, seed=42)

model_name = "Llama3-8b-Instruct"

with open("dump.jsonl", 'a') as out:
    for i in tqdm(range(len(dd))):
        t = "Create a question based on the following context: {}".format(' '.join((dd[i]['premise'], dd[i]['hypothesis'])))
        mess = [{"role": "system", "content": ""},
                {"role": "user", "content": t}]
        q = m.create_chat_completion(mess, max_tokens=250)["choices"][0]['message']['content']

        out.write(
            json.dumps({"uid": dd[i]["uid"], "statement": dd[i]["premise"], "contradicting statement": dd[i]["hypothesis"],
                        "question": q, "model": model_name, "label": "contradiction"}, ensure_ascii=False) + '\n') 

records = []
with open() as inp:
    for line in inp:
        records.append(json.loads(line.strip()))
df = pd.Dataframe.from_records(records)
df.to_csv(f"anli_train-r2_{model_name.lower()}.csv", index=None)