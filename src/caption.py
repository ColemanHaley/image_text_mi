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
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from dataset import COCODataset

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

LEN_CAPTIONING_PREFIX = 6  # TODO: compute prefix length

traceback.install()


def predict_step(*, caption_model, text_model, tokenizer, batch):

    device = caption_model.device
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # the conditional probability of the caption given the image
    loss_caption = caption_model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        labels=input_ids,
        attention_mask=attention_mask,
    )
    loss_text = text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    tokens = []
    cap_xent = []
    text_xent = []
    cap_ent = []
    text_ent = []

    batch_size = loss_caption.logits.shape[0]
    false_prefix = torch.zeros(LEN_CAPTIONING_PREFIX, dtype=torch.bool).to(device)

    for sentence in range(batch_size):
        mask = torch.cat([false_prefix, (attention_mask[sentence] == 1)])[
            :-LEN_CAPTIONING_PREFIX
        ]
        tokens.append(input_ids[sentence][mask])

        # compute cross-entropy for conditional(captioning) and prior(text) distributions
        cap_xent_sent = torch.nn.functional.cross_entropy(
            loss_caption.logits[sentence][mask],
            input_ids[sentence][mask],
            reduction="none",
        )
        text_xent_sent = torch.nn.functional.cross_entropy(
            loss_text.logits[sentence][mask],
            input_ids[sentence][mask],
            reduction="none",
        )
        # compute entropy for conditional and prior distributions
        cap_ent_sent = Categorical(logits=loss_caption.logits[sentence][mask]).entropy()
        text_ent_sent = Categorical(logits=loss_text.logits[sentence][mask]).entropy()

        cap_xent.extend(cap_xent_sent.cpu().tolist())
        text_xent.extend(text_xent_sent.cpu().tolist())
        cap_ent.extend(cap_ent_sent.cpu().tolist())
        text_ent.extend(text_ent_sent.cpu().tolist())

    tokens = [tokenizer.convert_ids_to_tokens(s) for s in tokens]

    index = [len(sent) * [i] for i, sent in enumerate(tokens)]

    return {
        "token": list(flatten(tokens)),
        "cap_xent": cap_xent,
        "text_xent": text_xent,
        "cap_ent": cap_ent,
        "text_ent": text_ent,
        "mutual_information": np.array(text_xent) - np.array(cap_xent),
        "sentence": list(flatten(index)),
    }


def prepare_batch(batch, processor, prefix="Caption the image in English."):
    images, captions, image_paths = zip(*batch)
    captions = [f"{prefix} {caption}" for caption in captions]
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
    if cfg.name == "mblip-bloomz":
        if cfg.quant:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_compute_dtype": torch.bfloat16,
            })

        model = Blip2ForConditionalGeneration.from_pretrained(
            cfg.path, **kwargs
        )

        processor = Blip2Processor.from_pretrained(
            cfg.path,
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not implemented.")
    return model, processor

def get_data(cfg, processor):
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
    return data

@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):

    caption_model, processor = load_model(cfg.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    caption_model.eval().to(device)
    text_model = caption_model.language_model

    data = get_data(cfg, processor)

    full_results = []
    for batch in tqdm(data):
        with torch.no_grad():
            print(batch)
            results = predict_step(
                caption_model=caption_model,
                text_model=text_model,
                tokenizer=processor.tokenizer,
                batch=batch,
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


if __name__ == "__main__":
    main()
