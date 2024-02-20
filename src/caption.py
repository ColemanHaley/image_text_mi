import logging

from dataset import COCODataset

import numpy as np
import pandas as pd
import torch
from pandas.core.common import flatten
from rich.logging import RichHandler
from rich import traceback
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

caption_model = Blip2ForConditionalGeneration.from_pretrained("Gregor/mblip-mt0-xl")

feature_extractor = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
tokenizer = AutoTokenizer.from_pretrained(
    "Gregor/mblip-mt0-xl",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
caption_model.eval().to(device)
text_model = caption_model.language_model


def predict_step(*, pixel_values, input_ids, attention_mask):

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
    for sentence in range(batch_size):
        mask = attention_mask[sentence] == 1
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

    tokens = [feature_extractor.tokenizer.convert_ids_to_tokens(s) for s in tokens]

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


# captions = [
#     "A test sentence.",
#     "A test sentence with indubitably obscure verbage.",
#     "Two dogs and a cat.",
#     "Two cats and a dog.",
# ]
# image_paths = [
#     "istockphoto-1251352680-612x612.jpg",
#     "istockphoto-1251352680-612x612.jpg",
#     "istockphoto-1251352680-612x612.jpg",
#     "istockphoto-1251352680-612x612.jpg",
# ]


def prepare_batch_mt0(batch, prefix="Caption the image in English."):
    images, captions, image_paths = zip(*batch)
    captions = [f"{prefix} {caption}" for caption in captions]
    batch = feature_extractor(
        images=images,
        text=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    batch.update({"image_paths": image_paths, "captions": captions})
    return batch


if __name__ == "__main__":
    val_dataloader = DataLoader(
        COCODataset(
            "data/coco",
            "val",
        ),
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=prepare_batch_mt0,
    )
    full_results = []
    for batch in tqdm(val_dataloader):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            results = predict_step(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        results.update(
            {
                "image_id": [batch["image_paths"][i] for i in results["sentence"]],
                "caption": [batch["captions"][i] for i in results["sentence"]],
            }
        )
        full_results.append(pd.DataFrame(results))

    with open("results_tok_mt0.csv", "w") as f:
        pd.concat(full_results, ignore_index=True).to_csv(f)
