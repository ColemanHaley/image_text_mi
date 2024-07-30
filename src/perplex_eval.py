from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import COCO35TextDataset


data_val = COCO35TextDataset( './data/coco/', 'dev')
def prepare_batch(captions, processor):

    batch = processor(
        text=captions,
        text_target=captions,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    return batch

processor = AutoProcessor.from_pretrained('google/paligemma-3b-pt-224')
processor.tokenizer.padding_side="right"
dataloader = DataLoader(data_val, batch_size=32, num_workers=4, shuffle=False, collate_fn=lambda b: prepare_batch(b, processor.tokenizer))

def eval(model_str, adapter=None):

    model = AutoModelForCausalLM.from_pretrained(model_str)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, "chaley22/gemma-captioning-lora")
    model.eval()
    model.to('cuda:0')
    loss = 0
    count = 0
    for batch in tqdm(dataloader):
        batch.to('cuda:0')
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss
        print(outputs.loss)
        count += 1
    return loss / count
print(f"PERPLEIXTY GEMMA: {eval('google/gemma-2b')}")
print(f"PERPLEXITY_FINETUNED_LORA: {eval('google/gemma-2b', adapter='chaley22/pali-captioning-lm2')}")
print(f"PERPLEIXTY FINETUNED: {eval('chaley22/pali-captioning-lm-nolora')}")
