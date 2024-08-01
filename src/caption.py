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
from torch.distributions import Categorical
from torch.utils.data import DataLoader

# from tqdm.rich import tqdm
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from dataset import COCODataset, COCO35Dataset, XM3600Dataset

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

IMAGE_TOKEN_ID = 257152


def get_mask_paligemma(sentence, input_ids, attention_mask, prefix_len):
    img = (input_ids[sentence] == IMAGE_TOKEN_ID).sum().item() 
    pad =  abs(
        attention_mask[sentence].sum().item() - input_ids.size(1)
    )
    

    mask = torch.zeros(input_ids.size(1), dtype=torch.bool)
    mask[img + (prefix_len+1):] = True
    mask[-pad:] = False
    return mask
    

def get_logprobs(logits, input_ids, mask_fn, tokenizer):
    batch_size = logits.shape[0]
    results = []
    tokens = []
    device = logits.device
    for i in range(batch_size):
        mask = mask_fn(i).to(device)
        labels = input_ids[i][mask]
        tokens.append(labels)
        # print(tokenizer.batch_decode(input_ids[i]))
        # print(tokenizer.batch_decode(input_ids[i][:-1][mask[1:]]))
        # print(tokenizer.batch_decode(input_ids[i][mask]))
        # print(logits.size())
        # print(input_ids.size())

       # xent = torch.nn.functional.cross_entropy(logits[i][mask], labels, reduction="none").cpu().tolist()
        logprobs = torch.nn.functional.log_softmax(logits[i][:-1][mask[1:]], dim=-1)
        topk = torch.topk(logprobs, 10)
        # print(tokenizer.batch_decode(topk.indices))
        # print(topk.values)
        # break
        xent = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze()
        for j, token_id in enumerate(labels):
            correction = correct_for_spaces(token_id, logprobs[j])
            if j > 0:
                xent[j-1] -= correction
            xent[j] += correction

        results.extend((-xent.cpu()).tolist())
    return results, tokens

def correct_for_spaces(token_id, logprobs):
    correction = 0
    # if token_id in whitespace_tokens:
    #     correction = torch.logsumexp(logprobs[whitespace_tokens])
    return correction


def predict_step(caption_model, text_model, tokenizer, batch, prefix_len):
    tokenizer, tokenizer_txt = tokenizer
    device = caption_model.device
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sents = tokenizer.batch_decode(input_ids)

    # the conditional probability of the caption given the image
    logits_cap = caption_model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    def mask_paligemma(x):
        return get_mask_paligemma(x, input_ids, attention_mask, prefix_len)

    # logits_cap = logits_cap[..., -input_ids.size(1) : -1, :].contiguous()

    cap_probs, cap_labels = get_logprobs(logits_cap, input_ids, mask_paligemma, tokenizer)

    captions = [c+"\n" for c in batch.captions]
    batch = tokenizer_txt(captions, return_tensors='pt', padding="longest", add_special_tokens=True)
    input_ids = batch["input_ids"].to(device)

    attention_mask = batch["attention_mask"].to(device)

    logits_txt = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    # logits_txt = logits_txt[..., :-1, :].contiguous()

    def mask_paligemma(x):
        return get_mask_paligemma(x, input_ids, attention_mask, prefix_len)


    txt_probs, txt_labels = get_logprobs(logits_txt, input_ids, mask_paligemma, tokenizer)
    txt_labels = [tokenizer.batch_decode(s) for s in txt_labels]
    cap_labels = [tokenizer.batch_decode(s) for s in cap_labels]
    # print(txt_labels)
    # print(cap_labels)
    assert txt_labels == cap_labels


    index = [len(sent) * [i] for i, sent in enumerate(cap_labels)]
    # print(list(zip(list(flatten(cap_labels)), list(flatten(txt_probs)), list(flatten(cap_probs)))))

    data = {
        "token": list(flatten(cap_labels)),
        "cap_xent":list(flatten(cap_probs)),
        "txt_xent":list(flatten(txt_probs)),
        "mutual_information": np.array(txt_probs) - np.array(cap_probs),
        "sentence": list(flatten(index)),
    }
    #print(pd.DataFrame(data))
    return data


def prepare_batch(batch, processor, prefix="Caption the image in English."):
    images, captions, image_paths = zip(*batch)
    captions = [f"{prefix}{caption}" for caption in captions]
    batch = processor(
        images=images,
        text=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
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
        )
    else:
        raise ValueError(f"Model {cfg.name} not implemented.")
    return model, processor


def get_data(cfg, processor):
    # compute prefix length
    prefix = f"caption {cfg.lang}\n"
    prefix_len = len(processor.tokenizer(prefix)["input_ids"])

    if cfg.dataset.name == "coco":
        data = DataLoader(
            COCODataset(
                cfg.dataset.path,
                cfg.dataset.split,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: prepare_batch(b, processor),
        )
    elif cfg.dataset.name == "xm3600":
        data = DataLoader(
            XM3600Dataset(
                cfg.dataset.path,
                cfg.lang,
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: prepare_batch(b, processor, prefix=prefix),
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
            collate_fn=lambda b: prepare_batch(b, processor, prefix=prefix),
        )
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
        data = [prepare_batch(zip(*(images, captions, image_paths)), processor)]
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not implemented.")
    return data, prefix_len


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.model.path,
        padding_side='right',
    )
    data = get_data(cfg, processor)

    caption_model, processor = load_model(cfg.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    caption_model.eval().to(device)
    #text_model = caption_model.language_model
    text_model = AutoModelForCausalLM.from_pretrained('chaley22/pali-captioning-lm-nolora')
    # text_model = PeftModel.from_pretrained(text_model, "chaley22/gemma-captioning-lora")
    text_model.eval().to(device)
    # tokenizer = AutoProcessor.from_pretrained('google/paligemma-3b-pt-224').tokenizer
    # tokenizer.padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b', padding_side="right")
    num = tokenizer.add_special_tokens({"eos_token": "\n"})
    processor.tokenizer.padding_side = "right"
    # tokenizer = processor.tokenizer


    data, prefix_len = get_data(cfg, processor)

    full_results = []
    for batch in tqdm(data):
        with torch.no_grad():
            results = predict_step(
                caption_model=caption_model,
                tokenizer=(processor.tokenizer, tokenizer),
                text_model=text_model,
                batch=batch,
                prefix_len=prefix_len,
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
