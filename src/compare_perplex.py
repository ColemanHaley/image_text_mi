import pandas as pd

df_gemma = pd.read_csv("outputs/multi30k/gemma-2b/results_ar.csv")
df_pali = pd.read_csv("outputs/multi30k/paligemma/results_ar.csv")
df_ft = pd.read_csv("outputs/multi30k/ft-pali/results_ar.csv")

# df_gemma = (
#     df_gemma.groupby("sentence").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
# )
# df_pali = df_pali.groupby("sentence").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
# df_ft = df_ft.groupby("sentence").apply(lambda x: x.iloc[1:]).reset_index(drop=True)
df_gemma = df_gemma[~df_gemma["token"].isin(["<eos>", ",", ".", "?", "!"])]
df_pali = df_pali[~df_pali["token"].isin(["<eos>", ",", ".", "?", "!"])]
df_ft = df_ft[~df_ft["token"].isin(["<eos>", ",", ".", "?", "!"])]
print("AVG_GEMMA:", df_gemma["gemma-2b_surprisal"].mean())
print("AVG_PALI:", df_pali["paligemma_surprisal"].mean())
print("AVG_FT:", df_ft["ft-pali_surprisal"].mean())
