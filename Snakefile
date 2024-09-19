LANGS = [
  'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa',
  'fi', 'fil', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
  'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
  'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
] # Drop quz b/c of COCO. 
LANGS_MULTI30K = ['ar', 'cs', 'de', 'en', 'fr']
DATASETS = ['xm3600', 'coco35', 'multi30k', 'multi30k_train']
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
  input:
    lambda wildcards: [ancient(f) for f in getattr(rules, wildcards.dataset).input],
     "src/caption.py"
  output:
      "outputs/{dataset}/{model}/results_{lang}.csv"
  wildcard_constraints:
    lang="|".join(LANGS),
    dataset="|".join(DATASETS),
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

rule pos:
  input:
    "outputs/{dataset}/paligemma/results_{lang}.csv",
    "src/caption.py"
  output:
    "outputs/{dataset}/pos/results_{lang}.csv"
  wildcard_constraints:
    dataset="|".join(DATASETS),
    lang="|".join(LANGS)
  shell:
    "python src/pos.py {wildcards.lang} {input} > {output}"

rule lang:
  input:
    expand("outputs/{dataset}/{model}/results_{{lang}}.csv", dataset=DATASETS, model=["ft-pali", "pos", "gemma-2b"])
  output:
    touch("outputs/{lang}.done")

rule combine_multi30k:
  input:
    "outputs/multi30k/pos/results_{lang}.csv",
    "outputs/multi30k/ft-pali/results_{lang}.csv",
    "src/combine.py"
  output:
    "outputs/results_{lang}_multi30k_tagged.csv"
  shell:
    """
    python src/combine.py \
    --lang {wildcards.lang} \
    --model ft-pali \
    --output outputs/results_{wildcards.lang}_multi30k_tagged.csv 
    """

rule combine_multi30k_gemma:
  input:
    "outputs/multi30k/pos/results_{lang}.csv",
    "outputs/multi30k/gemma-2b/results_{lang}.csv",
    "src/combine.py"
  output:
    "outputs/results_{lang}_multi30k_gemma_tagged.csv"
  shell:
    """
    python src/combine.py \
    --lang {wildcards.lang} \
    --model gemma-2b \
    --output outputs/results_{wildcards.lang}_multi30k_gemma_tagged.csv 
    """
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
