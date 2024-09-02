from PIL import Image
import torch


class WhitespaceCorrector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # all tokens that start with whitespace
        self.whitespace_tokens = set([
            token_id
            for token_id in range(tokenizer.vocab_size)
            if tokenizer.decode(token_id).lstrip() != tokenizer.decode(token_id)
        ])

    def correct_for_spaces(self, token_id, logprobs):
        correction = 0
        if token_id in self.whitespace_tokens:
            correction = torch.logsumexp(logprobs[self.whitespace_tokens], 0, False)
        return correction


def make_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert(mode="RGB")
    return img


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
    cs = [chr(n) for n in cs]
    return dict(zip())


byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# def fix_tokens(toks):
#     fixed = []
#     for token in toks:
#         token = bytearray(byte_decode(token))
#         try:
#             fixed.append(token.decode('utf-8'))
#         except UnicodeDecodingError:
#             fixed.append()
