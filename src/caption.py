import logging
import os
from pathlib import Path

import pdb
import hydra
import pandas as pd
import torch
import torch.nn.functional as F
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
    PaliGemmaProcessor,
)
from dataset import Multi30kDataset, COCO35Dataset, XM3600Dataset
from utils import WhitespaceCorrector, renumber_and_join_sents

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

IMAGE_TOKEN_ID = 257152


def get_mask_paligemma(sentence, input_ids, attention_mask, prefix_len):
    img = (input_ids[sentence] == IMAGE_TOKEN_ID).sum().item()
    if img > 0:
        img += 2
    pad = abs(attention_mask[sentence].sum().item() - input_ids.size(1))

    mask = torch.zeros(input_ids.size(1), dtype=torch.bool)
    mask[img + (prefix_len) :] = True
    if pad > 0:
        mask[-pad:] = False
    return mask


def get_logprobs(logits, input_ids, mask_fn, corrector):
    batch_size = logits.shape[0]
    results = []
    tokens = []
    device = logits.device
    for i in range(batch_size):
        # pdb.set_trace()
        mask = mask_fn(i).to(device)
        # print(mask)
        # print(input_ids[i])
        labels = input_ids[i][mask]
        tokens.append(labels)
        remember_me = mask[0]
        mask = torch.roll(mask, -1)
        mask[0] = remember_me

        logprobs = torch.nn.functional.log_softmax(logits[i][mask], dim=-1)
        # print(logprobs.shape)
        xent = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
        # pdb.set_trace()

        for j, token_id in enumerate(labels):
            correction = corrector.correct_for_spaces(token_id.item(), logprobs[j])
            # print(correction)
            # print(token_id.item())

            # if correction != 0:
            #     print('hi!!')
            if j > 0:
                xent[j - 1] += correction
            if len(xent) == 1:
                xent -= correction
            else:
                xent[j] -= correction

        results.extend((-xent.cpu()).tolist())
    return results, tokens


def predict_step(model, batch, tokenizer, prefix_len, corrector):
    device = model.device
    params = {}
    if "pixel_values" in batch:
        params["pixel_values"] = batch["pixel_values"].to(device)
        params["token_type_ids"] = batch["token_type_ids"].to(device)
        params["labels"] = batch["labels"].to(device)
    params["input_ids"] = batch["input_ids"].to(device)
    params["attention_mask"] = batch["attention_mask"].to(device)

    # the conditional probability of the caption given the image
    logits = model(**params).logits

    probs, labels = get_logprobs(
        logits,
        params["input_ids"],
        lambda s: get_mask_paligemma(
            s, params["input_ids"], params["attention_mask"], prefix_len
        ),
        corrector,
    )

    labels = [tokenizer.batch_decode(s) for s in labels]
    index = [len(sent) * [i] for i, sent in enumerate(labels)]

    data = {
        "token": list(flatten(labels)),
        "surprisal": list(flatten(probs)),
        "sentence": list(flatten(index)),
    }
    # print(pd.DataFrame(data))
    return data


def prepare_batch(batch, processor, prefix="Caption the image in English."):
    images, captions, image_paths = zip(*batch)

    if isinstance(processor, PaliGemmaProcessor):
        prefix = prefix.strip()
        batch = processor(
            images=images,
            text=[prefix] * len(captions),
            suffix=captions,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        captions = [f"{prefix}{caption}" for caption in captions]
    else:
        captions = [f"{prefix}{caption}" for caption in captions]
        batch = processor(
            captions, return_tensors="pt", padding="longest", add_special_tokens=True
        )
    batch.update({"image_paths": image_paths, "captions": captions})
    return batch


def load_model(cfg):
    kwargs = {}
    if cfg.quant:
        kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
    if cfg.name in ["paligemma"]:  # "mblip-bloomz", "mblip-mt0"]:
        model = PaliGemmaForConditionalGeneration.from_pretrained(cfg.path, **kwargs)

        processor = AutoProcessor.from_pretrained(
            cfg.path,
            padding_side="right",
        )
        processor.tokenizer.padding_side = "right"
    elif cfg.name in ["gemma-2b", "ft-pali"]:
        model = AutoModelForCausalLM.from_pretrained(cfg.path)
        processor = AutoTokenizer.from_pretrained(cfg.tok_path, padding_side="right")
        processor.add_eos_token = True
        # processor.add_special_tokens({"eos_token": "\n"})
    else:
        raise ValueError(f"Model {cfg.name} not implemented.")
    return model, processor


def get_data(cfg, processor, tokenizer):
    # compute prefix length
    prefix = f"caption {cfg.lang}\n"
    prefix_len = len(tokenizer(prefix)["input_ids"]) - 1

    if cfg.dataset.name == "test":
        # images = [Image.open("data/multi30k/images/1001465944.jpg")]
        images = [Image.open("data/multi30k/images/227689211.jpg")]
        images += [Image.open(cfg.dataset.path)] * 4 + [Image.open("test2.jpg")]
        captions = [
            # "A woman with a large purse is walking by a gate.",
            "A police officer in his uniform wearing an ear piece.",
            "A test sentence.",
            "A bicycle replica with a clock as the front wheel.",
            "A cat is laying on top of a dryer.",
            "A test sentence with indubitably obscure verbage.",
            "Two dogs and a cat.",
            "Two cats and a dog.",
            "the the the the the." "A bicycle replica with a clock as the front wheel.",
        ]
        image_paths = (
            ["data/multi30k/1001465944.jpg"] + [cfg.dataset.path] * 4 + ["test2.jpg"]
        )
        data = [
            prepare_batch(
                zip(*(images, captions, image_paths)), processor, prefix=prefix
            )
        ]
        return data, prefix_len
    elif cfg.dataset.name == "xm3600":
        dataset_class = XM3600Dataset
    elif cfg.dataset.name == "coco35":
        dataset_class = COCO35Dataset
    elif cfg.dataset.name == "multi30k":
        dataset_class = Multi30kDataset
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not implemented.")
    data = DataLoader(
        dataset_class(
            cfg.dataset.path,
            cfg.dataset.split,
            cfg.lang,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: prepare_batch(b, processor, prefix=prefix),
    )
    return data, prefix_len


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, processor = load_model(cfg.model)
    model.eval().to(device)

    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor

    data, prefix_len = get_data(cfg, processor, tokenizer)
    corrector = WhitespaceCorrector(tokenizer)

    full_results = []

    # pdb.set_trace()
    for batch in tqdm(data):
        with torch.no_grad():
            results = predict_step(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                prefix_len=prefix_len,
                corrector=corrector,
            )
        results.update(
            {
                "image_id": [batch["image_paths"][i] for i in results["sentence"]],
                "caption": [batch["captions"][i] for i in results["sentence"]],
            }
        )
        full_results.append(pd.DataFrame(results))

    full_results = pd.concat(full_results, ignore_index=True)

    full_results["token"] = full_results["token"].astype(str)
    sentence, caption, start_chars = renumber_and_join_sents(
        full_results["sentence"], full_results["token"]
    )
    full_results["caption"] = caption
    full_results["sentence"] = sentence
    full_results["start_char"] = start_chars
    full_results[cfg.model.name + "_surprisal"] = full_results["surprisal"]
    full_results.drop(columns=["surprisal"], inplace=True)

    hydra_output = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_file = f"{hydra_output}/../../../{cfg.out_file}"
    # add a date and time column
    full_results["date"] = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M")
    with open(out_file, "w") as f:
        full_results.to_csv(f, index=False)
    # os.symlink(
    #     f"{hydra_output}/{out_file.name}", f""
    # )


if __name__ == "__main__":
    main()
