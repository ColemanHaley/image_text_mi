from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from caption import load_model


def main(): 
    model_path = "google/paligemma-3b-ft-coco35l-224" 
    processor = AutoProcessor.from_pretrained(
        model_path
    )
    caption_model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path
    )


tokens = {
    0: "I",
    1: " am",
    2: " a",
    3: " mat",
    4: "ron",
    5: " in",
}

labels1 = [0,1,2,3,4]
labels2 = [0,1,2,3,5]
logits1 = [[
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 5],
    [0, 0, 1, 0, 1],
]]

def no_mask(idx):
    return torch.ones(5, dtype=torch.bool)
get_logpotbs(logits, input_ids, no_mask)
