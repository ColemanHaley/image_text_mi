import sys

import stanza
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv(sys.argv[2], index_col=0)

df["token"] = df["token"].astype(str)
df["caption"] = caption
df["sentence"] = sentence
df["start_char"] = start_chars
df["POS"] = "SKIP"
df["word_stanza"] = "#SKIP#"
# print(df)

df_sent = df.groupby("sentence").first().reset_index()[["caption", "sentence"]]
try:
    nlp = stanza.Pipeline(sys.argv[1], processors="tokenize,mwt,pos")
except Exception:
    nlp = stanza.Pipeline(sys.argv[1], processors="tokenize,pos")

documents = df_sent["caption"].tolist()
in_docs = [stanza.Document([], text=doc) for doc in documents]
out_docs = nlp(in_docs)
for i, doc in enumerate(out_docs):
    cap_info = df[df["sentence"] == i]
    for sent in doc.sentences:
        blacklist = []
        for word in sent.words:
            if isinstance(word.id, list):
                blacklist.extend(word.id)
            elif word.id in blacklist:
                continue
            else:
                first_tok = cap_info[cap_info["start_char"] == word.start_char]
                if len(first_tok) > 0:
                    idx = first_tok.index.astype(int)[0]
                    first_tok = first_tok.iloc[0]
                    idxs = [idx]

                    while (
                        idx < cap_info.index.astype(int)[-1]
                        and cap_info.loc[idx, "start_char"]
                        + len(cap_info.loc[idx, "token"])
                        <= word.end_char
                    ):
                        idxs.append(idx)
                        idx += 1

                    if (
                        cap_info.loc[idxs[-1], "start_char"]
                        + len(cap_info.loc[idxs[-1], "token"].strip())
                        == word.end_char
                    ):
                        df.loc[idxs, "POS"] = word.upos
                        df.loc[idxs, "word_stanza"] = word.text
                else:
                    continue


def sent_idx(sents):
    count = 0
    new_col = []
    for sent1, sent2 in zip(sents, sents[1:]):
        new_col.append(count)
        if sent2 != sent1:
            count = 0
        else:
            count += 1
    new_col.append(count)
    return new_col


df["sentence_idx"] = sent_idx(df["sentence"])


print(df.to_csv())
exit()


tagged = [
    (word, word.upos)
    for doc in out_docs
    for sent in doc.sentences
    for word in sent.words
]
