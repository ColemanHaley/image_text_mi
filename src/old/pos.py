import hydra
import pandas as pd
from rich import traceback
from tqdm import tqdm
import torch
from sft import SFT
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

traceback.install()
labels = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
id2label = {i: l for i, l in enumerate(labels)}


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    config = AutoConfig.from_pretrained(cfg.pos.model, num_labels=cfg.pos.num_tags)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.pos.model, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.pos.model)

    df = pd.read_csv("outputs/2024-02-26/16-04-51/results.csv")

    lang_sft = SFT(f"cambridgeltl/mbert-lang-sft-{cfg.language}-small")
    task_sft = SFT(f"cambridgeltl/mbert-task-sft-pos")

    lang_sft.apply(model, with_abs=False)
    task_sft.apply(model)
    df["caption"] = df.caption.str.lstrip("Caption the image in English.")
    captions = df.caption.unique()
    for caption in tqdm(captions):
        inputs = tokenizer(caption, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_token_class = [id2label[t.item()] for t in predictions[0]]
        pos_sent = {}
        for word_id, pos in zip(inputs.word_ids(), predicted_token_class):
            if word_id is not None:
                if word_id not in pos_sent:
                    pos_sent[word_id] = []
                pos_sent[word_id].append(pos)
        for key in pos_sent:
            if len(set(pos_sent[key])) > 1:
                print(caption)
                print(tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
                print(inputs.word_ids())
                print(predicted_token_class)
                break


if __name__ == "__main__":
    main()
