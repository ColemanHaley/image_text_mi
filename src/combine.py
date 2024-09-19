import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--model", type=str, default="ft-pali")
parser.add_argument("--output", type=str)
args = parser.parse_args()

df_cap = pd.read_csv(f"outputs/multi30k/pos/results_{args.lang}.csv")
df_txt = pd.read_csv(f"outputs/multi30k/{args.model}/results_{args.lang}.csv")

# remove lines where the token is <eos>
df_cap = df_cap[df_cap.token != "<eos>"]
# make index match
df_cap = df_cap.reset_index(drop=True)

df_cap[f"{args.model}_surprisal"] = df_txt[f"{args.model}_surprisal"]
df_cap["mutual_information"] = (
    df_cap[f"{args.model}_surprisal"] - df_cap["paligemma_surprisal"]
)
df_cap.to_csv(args.output, index=False)
