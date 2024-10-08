import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from pandas.core.common import flatten
from PIL import Image
from rich import traceback
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from dataset import COCODataset, COCO35Dataset, XM3600Dataset
from utils import WhitespaceCorrector

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

IMAGE_TOKEN_ID = 257152


def get_mask_paligemma(sentence, input_ids, attention_mask, prefix_len):
    img = (input_ids[sentence] == IMAGE_TOKEN_ID).sum().item()
    pad = abs(attention_mask[sentence].sum().item() - input_ids.size(1))

    mask = torch.zeros(input_ids.size(1), dtype=torch.bool)
    mask[img + (prefix_len + 1) :] = True
    mask[-pad:] = False
    return mask


def get_logprobs(logits, input_ids, mask_fn, corrector):
    batch_size = logits.shape[0]
    results = []
    tokens = []
    device = logits.device
    for i in range(batch_size):
        mask = mask_fn(i).to(device)
        labels = input_ids[i][mask]
        tokens.append(labels)
        logprobs = torch.nn.functional.log_softmax(logits[i][:-1][mask[1:]], dim=-1)
        xent = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
        for j, token_id in enumerate(labels):
            correction = corrector.correct_for_spaces(token_id, logprobs[j])
            if j > 0:
                xent[j - 1] -= correction
            xent[j] += correction

        results.extend((-xent.cpu()).tolist())
    return results, tokens


def predict_step_inner(
    model, prefix_len, corrector, input_ids, attention_mask, pixel_values=None
):
    device = model.device
    params = {}
    if pixel_values is not None:
        params["pixel_values"] = pixel_values.to(device)
    params["input_ids"] = input_ids.to(device)
    params["attention_mask"] = attention_mask.to(device)

    # the conditional probability of the caption given the image
    logits = model(**params).logits

    probs, labels = get_logprobs(
        logits,
        params["input_ids"],
        lambda s: get_mask_paligemma(s, params["input_ids"], params["attention_mask"], prefix_len),
        corrector,
    )

    return probs, labels


def predict_step(caption_model, text_model, tokenizer, batch, prefix_len, corrector):
    tokenizer, tokenizer_txt = tokenizer
    corrector, corrector_txt = corrector
    cap_probs, cap_labels = predict_step_inner(
        caption_model,
        prefix_len,
        corrector,
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
    )

    txt_probs, txt_labels = predict_step_inner(
        text_model,
        prefix_len,
        corrector_txt,
        batch["inputs_txt"],
        batch["attention_mask_txt"],
    )

    txt_labels = [tokenizer_txt.batch_decode(s) for s in txt_labels]
    cap_labels = [tokenizer.batch_decode(s) for s in cap_labels]
    assert txt_labels == cap_labels

    index = [len(sent) * [i] for i, sent in enumerate(cap_labels)]

    data = {
        "token": list(flatten(cap_labels)),
        "cap_xent": list(flatten(cap_probs)),
        "txt_xent": list(flatten(txt_probs)),
        "mutual_information": np.array(txt_probs) - np.array(cap_probs),
        "sentence": list(flatten(index)),
    }
    return data


def prepare_batch(batch, processor, tokenizer, prefix="Caption the image in English."):
    images, captions, image_paths = zip(*batch)
    captions = [f"{prefix}{caption}" for caption in captions]
    batch = processor(
        images=images,
        text=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    captions = [caption + "\n" for caption in captions]
    txt_batch = tokenizer(
        captions, return_tensors="pt", padding="longest", add_special_tokens=True
    )
    batch.update(
        {
            "inputs_txt": txt_batch["input_ids"],
            "attention_mask_txt": txt_batch["attention_mask"],
        }
    )
    batch.update({"image_paths": image_paths, "captions": captions})
    return batch


def load_model(cfg):
    kwargs = {}
    if cfg.name in ["pali-gemma"]:  # "mblip-bloomz", "mblip-mt0"]:
        if cfg.quant:
            kwargs.update(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": False,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                }
            )

        model = PaliGemmaForConditionalGeneration.from_pretrained(cfg.path, **kwargs)

        processor = AutoProcessor.from_pretrained(
            cfg.path,
            padding_side="right",
        )
    else:
        raise ValueError(f"Model {cfg.name} not implemented.")
    return model, processor


def get_data(cfg, processor, tokenizer):
    # compute prefix length
    prefix = f"caption {cfg.lang}\n"
    prefix_len = len(processor.tokenizer(prefix)["input_ids"])

    if cfg.dataset.name == "xm3600":
        data = DataLoader(
            XM3600Dataset(
                cfg.dataset.path,
                cfg.lang,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: prepare_batch(b, processor, tokenizer, prefix=prefix),
        )
    elif cfg.dataset.name == "coco35":
        data = DataLoader(
            COCO35Dataset(
                cfg.dataset.path,
                cfg.dataset.split,
                cfg.lang,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: prepare_batch(b, processor, tokenizer, prefix=prefix),
        )
        print(len(data.dataset))
    elif cfg.dataset.name == "test":
        images = [Image.open(cfg.dataset.path)] * 4 + [Image.open("test2.jpg")]
        captions = [
            "A test sentence.",
            "A bicycle replica with a clock as the front wheel.",
            "A cat is laying on top of a dryer.",
            "A test sentence with indubitably obscure verbage.",
            "Two dogs and a cat.",
            "Two cats and a dog.",
            "the the the the the." "A bicycle replica with a clock as the front wheel.",
        ]
        image_paths = [cfg.dataset.path] * 4 + ["test2.jpg"]
        data = [
            prepare_batch(zip(*(images, captions, image_paths)), processor, tokenizer)
        ]
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not implemented.")
    return data, prefix_len


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    caption_model, processor = load_model(cfg.model)
    caption_model.eval().to(device)

    text_model = AutoModelForCausalLM.from_pretrained(
        cfg.txt_model
    )
    text_model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", padding_side="right")
    tokenizer.add_special_tokens({"eos_token": "\n"})
    processor.tokenizer.padding_side = "right"

    data, prefix_len = get_data(cfg, processor, tokenizer)

    full_results = []
    correctors = (
        WhitespaceCorrector(processor.tokenizer),
        WhitespaceCorrector(tokenizer),
    )
    for batch in tqdm(data):
        with torch.no_grad():
            results = predict_step(
                caption_model=caption_model,
                tokenizer=(processor.tokenizer, tokenizer),
                text_model=text_model,
                batch=batch,
                prefix_len=prefix_len,
                corrector=correctors,
            )
        results.update(
            {
                "image_id": [batch["image_paths"][i] for i in results["sentence"]],
                "caption": [batch["captions"][i] for i in results["sentence"]],
            }
        )
        full_results.append(pd.DataFrame(results))

    hydra_output = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{hydra_output}/{cfg.out_file}", "w") as f:
        pd.concat(full_results, ignore_index=True).to_csv(f)
    os.symlink(f"{hydra_output}/{cfg.out_file}", f"{hydra_output}/../../{cfg.out_file}")


if __name__ == "__main__":
    main()
