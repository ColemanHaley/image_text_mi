import sys
import torch
import torch.nn.functional as F
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils import WhitespaceCorrector
from PIL import Image


model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model.eval()
model.to("cuda")

cap_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-ft-coco35l-224"
)
processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-coco35l-224")
cap_model.eval()
cap_model.to("cuda")

corrector = WhitespaceCorrector(tokenizer)
cap = "a woman with a large purse is walking by a gate."
cap = "A police officer in his uniform wearing an ear piece."

toks = tokenizer(cap, return_tensors="pt")
toks = toks.to("cuda")
with torch.no_grad():
    logits = model(**toks).logits

    img = Image.open(sys.argv[2])
    img = img.convert("RGB")
    procs = processor(
        images=img,
        text="caption en",
        suffix=cap,
        return_tensors="pt",
    )
    procs = procs.to("cuda")
    import pdb

    pdb.set_trace()
    logits_cap = cap_model(**procs).logits
    print(len(procs.input_ids[0]))
    # assert logits_cap[:, -logits.shape[1] :, :].shape == logits.shape

    logprobs = F.log_softmax(logits, dim=-1)
    print(logprobs[0, 0, 0])
    print(F.softmax(logits, dim=-1))
    print(F.log_softmax(logits_cap[0, 0], dim=-1)[0])
    print(F.softmax(logits_cap[0, 0], dim=-1))
    # assert logprobs[0, 0, 0] == F.log_softmax(logits_cap[0, 0], dim=-1)[0]
    xent = torch.gather(logprobs, -1, toks.input_ids[:, 1:].unsqueeze(-1)).squeeze()
    print(xent)
    # xent = []
    for i in range(len(toks.input_ids[0]), 2, -1):
        print(toks.input_ids[0, -i + 1].item())
        # xent.append(logprobs[0, -i, toks.input_ids[0, -i + 1]])
        correction = corrector.correct_for_spaces(
            toks.input_ids[0, -i + 1].item(), logprobs[0, -i]
        )
        print(toks.input_ids[0, -i + 1].item())
        print(correction)
        if i < len(toks.input_ids[0]):
            xent[len(toks.input_ids[0]) - i - 1] -= correction
        xent[len(toks.input_ids[0]) - i] += correction

    labels = procs.input_ids[procs.token_type_ids == 1]
    logprobs = F.log_softmax(logits_cap, dim=-1)
    logprobs = logprobs[:, -len(labels) - 1 :]

    topk = torch.topk(logprobs[0, -len(toks.input_ids[0]) - 1 :, :], 10, dim=-1)
    print(list(zip(processor.tokenizer.batch_decode(topk.indices), topk.values)))
    print(logprobs.shape)
    print(labels.shape)
    print(procs)
    xent_cap = torch.gather(logprobs, -1, labels.unsqueeze(-1).unsqueeze(0)).squeeze()
    print(xent_cap)

    for i in range(len(toks.input_ids[0]), 2, -1):
        print(toks.input_ids[0, -i + 1].item())
        # xent_cap.append(logprobs[0, -i, toks.input_ids[0, -i + 1]])
        correction = corrector.correct_for_spaces(
            toks.input_ids[0, -i + 1].item(), logprobs[0, -i]
        )
        print(correction)
        if i < len(toks.input_ids[0]):
            xent_cap[len(toks.input_ids[0]) - i - 1] -= correction
        xent_cap[len(toks.input_ids[0]) - i] += correction
    for tok, x, x_cap in zip(toks.input_ids[0, 1:], xent, xent_cap):
        print(tokenizer.decode(tok), -x.item(), -x_cap.item())
    import pdb

    pdb.set_trace()
    procs2 = processor(images=img, text="caption fr", return_tensors="pt")
    procs2 = procs2.to("cuda")
    outputs = cap_model.generate(
        **procs2, max_new_tokens=100, return_dict_in_generate=True, output_scores=True
    )
    out = tokenizer.batch_decode(outputs[0])
    print(out)
    print(outputs[0])
    scores = torch.stack(outputs.scores, dim=1)
    scores = F.log_softmax(scores, dim=-1)
    logprobs = torch.gather(scores, -1, outputs[0][:, -scores.shape[1] :].unsqueeze(-1))
    toks2 = tokenizer(out, return_tensors="pt")
    toks2.to("cuda")
    logits = model(**toks2).logits
    logprobs_cap = F.log_softmax(logits, dim=-1)
    logprobs_cap = torch.gather(
        logprobs_cap, -1, toks2.input_ids.unsqueeze(-1)
    ).squeeze()
    for tok, x, x_cap in zip(toks2.input_ids[0, 1:], logprobs, logprobs_cap):
        print(tokenizer.decode(tok), x.item(), x_cap.item())
        # print(processor.tokenizer.decode(tok))
