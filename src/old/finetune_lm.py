import logging
import os

import hydra
import torch
import wandb
from peft import get_peft_model, LoraConfig
from rich import traceback
from rich.logging import RichHandler

from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)

from dataset import COCO35LMDataset
logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

os.environ["WANDB_PROJECT"] = 'paligemma-lm'

run = wandb.init()
i = 0
def prepare_batch(batch, processor, seqlens):
    images, captions, langs = zip(*batch)
    prefixes = [f"caption {lang}\n" for lang in langs]

    batch = processor(
        images=images,
        text=prefixes,
        suffix=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch.input_ids
    seqlens.append(input_ids.size(1))
    global i
    if i % 100 == 0 and i != 0:
        run.log({"max_seq_len": max(seqlens), "avg_seq_len": sum(seqlens)/len(seqlens), "min_seq_len": min(seqlens)}, step=i)
        seqlens = []
    i += 1
    return batch


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.model.path,
    )

    data = COCO35LMDataset(
        cfg.dataset.path,
        "train",
    )

    data_val = COCO35LMDataset(
        cfg.dataset.path,
        "dev",
    )


    seqlens = []


    model = PaliGemmaForConditionalGeneration.from_pretrained(cfg.model.path)

    processor = AutoProcessor.from_pretrained(
        cfg.model.path,
    )

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="./paligemma-lm-coco35/",
        num_train_epochs=1,
        #auto_find_batch_size=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        bf16=True,
        save_steps=100,
        push_to_hub=True,
        save_total_limit=1,
        report_to=["wandb"],
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        train_dataset=data,
        eval_dataset=data_val,
        data_collator=lambda b: prepare_batch(b, processor, seqlens),
        args=args,
    )
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
