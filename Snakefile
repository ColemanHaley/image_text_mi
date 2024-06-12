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
        directory(f"{SCRATCH}data/coco/{{split}}")
    wildcard_constraints:
        split="train2014|val2014|test2014|val2017"
    shell:
        f"""
        wget http://images.cocodataset.org/zips/{{wildcards.split}}.zip -P {SCRATCH}
        unzip {SCRATCH}{{wildcards.split}}.zip -d {SCRATCH}
        mkdir -p {SCRATCH}data/coco/{{wildcards.split}}
        mv {SCRATCH}{{wildcards.split}} {SCRATCH}data/coco/{{wildcards.split}}
        """
rule xm3600_images:
    output:
        directory(f"{SCRATCH}data/xm3600/images")
    shell:
        f"""
        mkdir -p {SCRATCH}data/xm3600/images
        wget https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz -P {SCRATCH}
        tar -xvzf {SCRATCH}images.tgz -C {SCRATCH}data/xm3600/images
        rm {SCRATCH}images.tgz
        """

rule xm3600_annotations:
  output:
    f"{SCRATCH}data/xm3600/captions.jsonl"
  shell:
    f"""
    mkdir -p {SCRATCH}data/xm3600/
    wget https://google.github.io/crossmodal-3600/web-data/captions.zip -P {SCRATCH}
    unzip {SCRATCH}captions.zip -d {SCRATCH}data/xm3600/
    """

rule coco_35_annotations:
    output:
        f"{SCRATCH}data/coco/annotations/{{split}}_35_caption.jsonl"
    shell:
        f"""
        wget https://storage.googleapis.com/crossmodal-3600/coco_mt_{{wildcards.split}}.jsonl.bz2 -P {SCRATCH}
        bunzip2 -f {SCRATCH}coco_mt_{{wildcards.split}}.jsonl.bz2 
        mkdir -p {SCRATCH}data/coco/annotations
        mv {SCRATCH}coco_mt_{{wildcards.split}}.jsonl {SCRATCH}data/coco/annotations/{{wildcards.split}}_35_caption.jsonl
        """

rule coco_annotations:
    output:
        f"{SCRATCH}data/coco/annotations/train_caption.json"
    shell:
        "wget https://huggingface.co/chaley22/coco_mblip/resolve/main/train_caption.json?download=true -o {output}"

rule coco_val:
    input:
        f"{SCRATCH}data/coco/val2014",
        f"{SCRATCH}data/coco/annotations/dev_35_caption.jsonl",
        f"{SCRATCH}data/coco/annotations/train_caption.json",
    shell:
      """
      ln -s {SCRATCH}data/ data
      """

rule xm3600:
  input:
    f"{SCRATCH}data/xm3600/images",
    f"{SCRATCH}data/xm3600/captions.jsonl"

rule caption_lang:
    input:
        f"{SCRATCH}data/coco/val2014",
        f"{SCRATCH}data/coco/annotations/dev_35_caption.jsonl",
    output:
        "outputs/results_{lang}.csv"
    wildcard_constraints:
      lang= "|".join(LANGS)
    shell:
      "python src/caption.py hydra.job.chdir=False lang={wildcards.lang} out_file=results_{wildcards.lang}.csv"

rule caption_3600:
  input:
    f"{SCRATCH}data/xm3600/images",
    f"{SCRATCH}data/xm3600/captions.jsonl"
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
    expand("outputs/results_{lang}.csv", lang=LANGS)

storage:
  provider="sharepoint"
  site-url="https://uoe.sharepoint.com/sites/Groundedness/"
  allow_overwrite=True


rule pos:
  input:
    storage("mssp://Results/results/{dir}/results_{spec}.csv")
  output:
