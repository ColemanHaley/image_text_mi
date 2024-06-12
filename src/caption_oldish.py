import logging

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
)

from dataset import COCODataset, COCO35Dataset

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

LEN_CAPTIONING_PREFIX = 2  # TODO: compute prefix length

traceback.install()


def predict_step(caption_model, text_model, tokenizer, batch, prefix_len):

    device = caption_model.device
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # the conditional probability of the caption given the image
    logits_cap = caption_model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits
    logits_txt = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    tokens = []
    cap_xent = []
    txt_xent = []
    cap_ent = []
    txt_ent = []

    batch_size = logits_cap.shape[0]
    #false_prefix = torch.zeros(prefix_len - 1, dtype=torch.bool).to(device)

    logits_cap = logits_cap[..., -input_ids.size(1) : -1, :].contiguous()
    logits_txt = logits_txt[..., :-1, :].contiguous()
    for sentence in range(batch_size):
        pad_len = abs(attention_mask[sentence].sum().item() - input_ids.size(1))

        mask = torch.zeros(input_ids.size(1) - 1, dtype=torch.bool).to(device)
        mask[pad_len + prefix_len - 1 :] = True
        labels = input_ids[sentence][1:][mask]
        tokens.append(labels)

        # compute cross-entropy for conditional(captioning) and prior(text) distributions
        cap_xent_sent = torch.nn.functional.cross_entropy(
            logits_cap[sentence][mask],
            labels,
            reduction="none",
        )
        txt_xent_sent = torch.nn.functional.cross_entropy(
            logits_txt[sentence][mask],
            labels,
            reduction="none",
        )
        # compute entropy for conditional and prior distributions
        cap_ent_sent = Categorical(logits=logits_cap[sentence][mask]).entropy()
        txt_ent_sent = Categorical(logits=logits_txt[sentence][mask]).entropy()

        cap_xent.extend(cap_xent_sent.cpu().tolist())
        txt_xent.extend(txt_xent_sent.cpu().tolist())
        cap_ent.extend(cap_ent_sent.cpu().tolist())
        txt_ent.extend(txt_ent_sent.cpu().tolist())

    tokens = [tokenizer.batch_decode(s) for s in tokens]

    index = [len(sent) * [i] for i, sent in enumerate(tokens)]

    return {
        "token": list(flatten(tokens)),
        "cap_xent": cap_xent,
        "txt_xent": txt_xent,
        "cap_ent": cap_ent,
        "txt_ent": txt_ent,
        "mutual_information": np.array(txt_xent) - np.array(cap_xent),
        "sentence": list(flatten(index)),
    }


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
    prefix = f"caption {cfg.lang}"
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
            # "A test sentence.",
            "A bicycle replica with a clock as the front wheel.",
            "A cat is laying on top of a dryer.",
            # "A test sentence with indubitably obscure verbage.",
            "Two dogs and a cat.",
            "Two cats and a dog.",
            "A bicycle replica with a clock as the front wheel.",
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
    )
    data = get_data(cfg, processor)

    caption_model, processor = load_model(cfg.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    caption_model.eval().to(device)
    text_model = caption_model.language_model

    data, prefix_len = get_data(cfg, processor)

    full_results = []
    for batch in tqdm(data):
        with torch.no_grad():
            results = predict_step(
                caption_model=caption_model,
                text_model=text_model,
                tokenizer=processor.tokenizer,
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
    # with open("outputs/{cfg.out_file}", "w") as f:
    #     pd.concat(full_results, ignore_index=True).to_csv(f)


if __name__ == "__main__":
    main()
