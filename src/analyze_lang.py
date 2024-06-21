import sys
import pandas
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv(sys.argv[1])

grps = (df.word_stanza != df.word_stanza.shift()).cumsum() # TODO: fix for repeated words
by_word =  df.groupby(grps)[['txt_xent', 'cap_xent', 'cap_ent', 'txt_ent', 'mutual_information']].progress_transform(lambda s: sum(s))
df = df[['sentence', 'POS', 'word_stanza', 'caption']].join(by_word)

df = df.groupby(grps).first().reset_index(drop=True)


