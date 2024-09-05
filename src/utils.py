from PIL import Image
import torch


class WhitespaceCorrector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # all tokens that start with whitespace
        self.whitespace_tokens = set(
            [
                token_id
                for token_id in range(tokenizer.vocab_size)
                if tokenizer.decode(token_id).lstrip() != tokenizer.decode(token_id)
            ]
        )

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


def renumber_and_join_sents(sents, tokens):
    # renumber the sentences, dealinging with the batch resetting index, and providing word broundary info
    total, count, start_char = 0, 0, 0
    sentence_ids, sentences, start_chars = [0], [], [0]
    sent_toks = []
    for i, (sent1, sent2) in enumerate(zip(sents, sents[1:])):
        n_spaces = len(tokens[i]) - len(tokens[i].lstrip())
        start_chars.append(start_char + n_spaces if count > 0 else 0)
        start_char = len(tokens[i].strip()) if count == 0 else len(tokens[i])
        sent_toks.append(tokens[i])

        count += 1

        if sent1 != sent2:
            sentences.extend(["".join(sent_toks).strip()] * count)
            total, count, start_char = total + 1, 0, 0
            sent_toks = []
        sentence_ids.append(total)

    # handle the last token
    sent_toks.append(tokens[i + 1])
    # start_chars.append(start_char + len(tokens[i + 1]) - len(tokens[i + 1].lstrip()))
    sentences.extend(["".join(sent_toks).strip()] * (count + 1))

    return sentence_ids, sentences, start_chars


# def fix_tokens(toks):
#     fixed = []
#     for token in toks:
#         token = bytearray(byte_decode(token))
#         try:
#             fixed.append(token.decode('utf-8'))
#         except UnicodeDecodingError:
#             fixed.append()
