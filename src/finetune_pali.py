import logging
import os
import math

import hydra
import torch
import wandb
from peft import get_peft_model, LoraConfig
from rich import traceback
from rich.logging import RichHandler
from PIL import Image
import torch.nn.functional as F

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

from dataset import COCO35TextDataset
from cached_model import PaliGemmaCached

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

traceback.install()

os.environ["WANDB_PROJECT"] = 'pali-captioning-lm'

run = wandb.init()

def prepare_batch(captions, processor):

    batch = processor.tokenizer(
        text=captions,
        text_target=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    # batch['labels'] = batch['input_ids'][:,1:]
    # print(batch['labels'].size())
    # print(batch['input_ids'].size())
    return batch

@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg):

    data = COCO35TextDataset(
        cfg.dataset.path,
        "train",
    )

    data_val = COCO35TextDataset(
        cfg.dataset.path,
        "dev_sub",
    )


    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b')


    processor = AutoProcessor.from_pretrained(
        cfg.model.path,
    )
    processor.tokenizer.padding_side="right"
    
    # batch = processor(
    #     images=[Image.open('data/coco/avg_train_224.jpg')]*32,
    #     text=["caption"]*32,
    #     padding="longest",
    #     truncation=True,
    #     return_tensors="pt",
    # )

    # past_kv = model(**batch)
    # cached_model = PaliGemmaCached.from_pretrained(model)
    # cached_model.cache_prefix(past_kv)
    cached_model = model

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(cached_model, lora_config)

    def compute_custom_metric(pred):
        logits = torch.from_numpy(pred.predictions)
        labels = torch.from_numpy(pred.label_ids)
        loss = F.cross_entropy(logits.view(-1, processor.tokenizer.vocab_size), labels.view(-1))
        return {'perplexity': math.exp(loss), 'calculated_loss': loss}

    args = TrainingArguments(
        output_dir="./pali-captioning-lm2/",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=20,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        eval_on_start=True,
        push_to_hub=True,
        save_total_limit=3,
        report_to=["wandb"],
        dataloader_pin_memory=True
    )
    trainer = Trainer(
        model=model,
        train_dataset=data,
        eval_dataset=data_val,
        compute_metrics=compute_custom_metric,
        data_collator=lambda b: prepare_batch(b, processor),
        args=args,
    )
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
