LANGS = [
  'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa',
  'fi', 'fil', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
  'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
  'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
] # Drop quz b/c of COCO. 
include: "./data.smk"

rule avg_img:
  input:
    "data/coco/train2017"
  output:
    "data/coco/avg_train_224.jpg"
  shell:
    "python src/average_image.py {input} {output} --size 224"
    

rule blank_coco35:
    output:
        "data/coco/blank.jpg"
    shell:
        "convert -size 224x224 xc:black {output}"


rule caption:
  #input:
      #    "data/{dataset}/complete"
    output:
        "outputs/{dataset}/{model}/results_{lang}.csv"
    wildcard_constraints:
      lang="|".join(LANGS),
      dataset="xm|coco|multi30k",
      model="gemma-2b|paligemma|ft-pali",
    shell:
      """
      mkdir -p outputs/{wildcards.dataset}/{wildcards.model}
      python src/caption.py \
      lang={wildcards.lang} \
      model={wildcards.model} \
      dataset={wildcards.dataset} \
      out_file={output}
      """

rule caption_multi30k_train:
    input:
        "data/multi30k/complete"
    output:
        "outputs/{dataset}/{model}/results_{lang}.csv"
    wildcard_constraints:
      lang="en|de|fr|ar|cs",
      dataset="multi30k_train",
      model="gemma-2b|paligemma|ft-pali",
    shell:
      """
      mkdir -p outputs/{wildcards.dataset}/{wildcards.model}
      python src/caption.py \
      lang={wildcards.lang} \
      model={wildcards.model} \
      dataset={wildcards.dataset} \
      out_file={output}
      """


rule all_3600:
  input:
    expand("outputs/results_{lang}_xm.csv", lang=LANGS)

rule all_gemma:
  input:
    expand("outputs/results_{lang}_{dataset}_gemma.csv", lang=LANGS, dataset=['xm', 'coco'])

rule all_coco:
  input:
    expand("outputs/results_{lang}_coco.csv", lang=LANGS)

rule pos:
  input:
    "outputs/{dataset}/paligemma/results_{lang}.csv"
  output:
    "outputs/{dataset}/pos/results_{lang}.csv"
  wildcard_constraints:
    dataset="xm|coco|multi30k|multi30k_train",
    lang="|".join(LANGS)
  shell:
    "python src/pos.py {wildcards.lang} {input} > {output}"

rule combine_multi30k:
  input:
    "outputs/multi30k/pos/results_{lang}.csv",
    "outputs/multi30k/ft-pali/results_{lang}.csv"
  output:
    "outputs/results_{lang}_multi30k_tagged.csv"
  run:
    import pandas as pd
    df_cap = pd.read_csv(f"outputs/multi30k/pos/results_{wildcards.lang}.csv")
    df_txt = pd.read_csv(f"outputs/multi30k/ft-pali/results_{wildcards.lang}.csv")
    print(df_txt.columns)

    df_cap["ft-pali_surprisal"] = df_txt["ft-pali_surprisal"]
    df_cap["mutual_information"] =   df_cap["ft-pali_surprisal"] - df_cap["paligemma_surprisal"]
    df_cap.to_csv(f"outputs/results_{wildcards.lang}_multi30k_tagged.csv", index=False)

rule combine_multi30k_gemma:
  input:
    "outputs/multi30k/pos/results_{lang}.csv",
    "outputs/multi30k/gemma-2b/results_{lang}.csv"
  output:
    "outputs/results_{lang}_multi30k_gemma_tagged.csv"
  run:
    import pandas as pd
    df_cap = pd.read_csv(f"outputs/multi30k/pos/results_{wildcards.lang}.csv")
    df_txt = pd.read_csv(f"outputs/multi30k/gemma-2b/results_{wildcards.lang}.csv")

    df_cap["gemma-2b_surprisal"] = df_txt["gemma-2b_surprisal"]
    df_cap["mutual_information"] =  df_cap["gemma-2b_surprisal"] - df_cap["paligemma_surprisal"] 
    df_cap.to_csv(f"outputs/results_{wildcards.lang}_multi30k_gemma_tagged.csv", index=False)
rule combine_multi30k_train:
  input:
    "outputs/multi30k_train/pos/results_{lang}.csv",
    "outputs/multi30k_train/ft-pali/results_{lang}.csv"
  output:
    "outputs/results_{lang}_multi30k_train_tagged.csv"
  run:
    import pandas as pd
    df_cap = pd.read_csv(f"outputs/multi30k_train/pos/results_{wildcards.lang}.csv")
    df_txt = pd.read_csv(f"outputs/multi30k_train/ft-pali/results_{wildcards.lang}.csv")
    print(df_txt.columns)

    df_cap["ft-pali_surprisal"] = df_txt["ft-pali_surprisal"]
    df_cap["mutual_information"] =   df_cap["ft-pali_surprisal"] - df_cap["paligemma_surprisal"]
    df_cap.to_csv(f"outputs/results_{wildcards.lang}_multi30k_train_tagged.csv", index=False)
rule combine_multi30k_gemma_train:
  input:
    "outputs/multi30k_train/pos/results_{lang}.csv",
    "outputs/multi30k_train/gemma-2b/results_{lang}.csv"
  output:
    "outputs/results_{lang}_multi30k_train_gemma_tagged.csv"
  run:
    import pandas as pd
    df_cap = pd.read_csv(f"outputs/multi30k_train/pos/results_{wildcards.lang}.csv")
    df_txt = pd.read_csv(f"outputs/multi30k_train/gemma-2b/results_{wildcards.lang}.csv")

    df_cap["gemma-2b_surprisal"] = df_txt["gemma-2b_surprisal"]
    df_cap["mutual_information"] =  df_cap["gemma-2b_surprisal"] - df_cap["paligemma_surprisal"] 
    df_cap.to_csv(f"outputs/results_{wildcards.lang}_multi30k_train_gemma_tagged.csv", index=False)
# rule download_3600:
#   output:
#       "outputs/results_{lang}_xm.csv"
#   shell:
#     """
#     huggingface-cli download chaley22/pali-captioning-lm-nolora --include 'outputs_xm/*/results_{wildcards.lang}_xm.csv' --local-dir outputs/
#     mv outputs/outputs_xm/*/results_{wildcards.lang}_xm.csv outputs/
#     """
# 
# ruleorder: download_3600 > caption_3600
