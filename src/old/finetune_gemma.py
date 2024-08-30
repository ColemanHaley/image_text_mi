import logging
import os

import hydra
import torch
import wandb
from peft import get_peft_model, LoraConfig
from rich import traceback
from rich.logging import RichHandler

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

from dataset import COCO35TextDataset
logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

os.environ["WANDB_PROJECT"] = 'gemma-captioning'

run = wandb.init()

def prepare_batch(captions, processor):

    batch = processor(
        text=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    batch['labels'] = batch['input_ids'][:,1:]
    print(batch['labels'].size())
    print(batch['input_ids'].size())
    return batch

@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):

    data = COCO35TextDataset(
        cfg.dataset.path,
        "train",
    )

    data_val = COCO35TextDataset(
        cfg.dataset.path,
        "dev",
    )


    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')
    past_kv = model(prefix)
    cached_model = CachedPaliGemma.from_pretrained(model)
    cached_model.cache_prefix(past_kv)

    processor = AutoProcessor.from_pretrained(
        cfg.model.path,
    ).tokenizer

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    args = TrainingArguments(
        output_dir="./gemma-captioning/",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=1000,
        push_to_hub=True,
        save_total_limit=3,
        report_to=["wandb"],
        dataloader_pin_memory=True

    )
    trainer = Trainer(
        model=model,
        train_dataset=data,
        data_collator=lambda b: prepare_batch(b, processor),
        args=args,
    )
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
