SCRATCH="/exports/eddie/scratch/s2189251/grounded/"
localrules: all_3600, all_coco
LANGS = [
  'ar', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa',
  'fi', 'fil', 'fr', 'he', 'hi', 'hr', 'hu', 'id', 'it',
  'ja', 'ko', 'mi', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
  'sv', 'sw', 'te', 'th', 'tr', 'uk', 'vi', 'zh'
] # Drop quz b/c of COCO. 

rule coco_images:
    output:
        directory("data/coco/{split}")
    wildcard_constraints:
        split="train2014|val2014|test2014|val2017"
    shell:
        """
        wget http://images.cocodataset.org/zips/{{wildcards.split}}.zip
        unzip {wildcards.split}.zip
        mkdir -p data/coco/{wildcards.split}
        mv {wildcards.split} data/coco/{wildcards.split}
        """

rule xm3600_images:
    output:
        directory("data/xm3600/images")
    shell:
        """
        mkdir -p data/xm3600/images
        wget https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz
        tar -xvzf {SCRATCH}images.tgz -C data/xm3600/images
        rm images.tgz
        """

rule xm3600_annotations:
  output:
    "data/xm3600/captions.jsonl"
  shell:
    """
    mkdir -p data/xm3600/
    wget https://google.github.io/crossmodal-3600/web-data/captions.zip
    unzip captions.zip -d data/xm3600/
    """

rule coco_35_annotations:
    output:
        "data/coco/annotations/{split}_35_caption.jsonl"
    shell:
        """
        wget https://storage.googleapis.com/crossmodal-3600/coco_mt_{wildcards.split}.jsonl.bz2 
        bunzip2 -f coco_mt_{wildcards.split}.jsonl.bz2 
        mkdir -p data/coco/annotations
        mv coco_mt_{wildcards.split}.jsonl {SCRATCH}data/coco/annotations/{wildcards.split}_35_caption.jsonl
        """

rule coco_annotations:
    output:
        f"data/coco/annotations/train_caption.json"
    shell:
        "wget https://huggingface.co/chaley22/coco_mblip/resolve/main/train_caption.json?download=true -o {output}"

rule coco_val:
    input:
        "data/coco/val2014",
        "data/coco/annotations/dev_35_caption.jsonl",
        "data/coco/annotations/train_caption.json",

rule xm3600:
  input:
    "data/xm3600/images",
    "data/xm3600/captions.jsonl"

rule caption_lang:
    input:
        "data/coco/val2014",
        "data/coco/annotations/dev_35_caption.jsonl",
    output:
        "outputs/results_{lang}_coco.csv"
    wildcard_constraints:
      lang= "|".join(LANGS)
    shell:
      "python src/caption.py hydra.job.chdir=False lang={wildcards.lang} out_file=results_{wildcards.lang}.csv"

rule caption_3600:
  input:
    f"data/xm3600/images",
    f"data/xm3600/captions.jsonl"
  output:
    "outputs/results_{lang}_xm.csv"
  wildcard_constraints:
    lang= "|".join(LANGS)
  shell:
    "python src/caption.py hydra.job.chdir=False lang={wildcards.lang} out_file=results_{wildcards.lang}_xm.csv dataset=xm3600"

rule all_3600:
  input:
    expand("outputs/results_{lang}_xm.csv", lang=LANGS)

rule all_coco:
  input:
    expand("outputs/results_{lang}_coco.csv", lang=LANGS)

rule pos:
  input:
    "outputs/results_{lang}_{file}.csv"
  output:
    "outputs/results_{lang}_{file}_tagged.csv"
  wildcard_constraints:
    file="coco|xm",
    lang="|".join(LANGS)
  shell:
    "python src/pos_stanza_new.py {wildcards.lang} {input} > {output}"

rule all_pos:
  input:
    expand("outputs/results_{lang}_{dataset}_tagged.csv", lang=LANGS, dataset=['xm', 'coco'])
