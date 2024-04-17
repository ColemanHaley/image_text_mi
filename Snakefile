SCRATCH="/exports/eddie/scratch/s2189251/grounded/"
rule coco_images:
    output:
        directory(f"{SCRATCH}data/coco/val2017/")
    shell:
        f"""
        wget http://images.cocodataset.org/zips/{{wildcards.split}}.zip -P {SCRATCH}
        unzip {SCRATCH}{{wildcards.split}}.zip -d {SCRATCH}
        mkdir -p {SCRATCH}data/coco/{{wildcards.split}}
        mv {SCRATCH}{{wildcards.split}} {SCRATCH}data/coco/{{wildcards.split}}/
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
        f"{SCRATCH}data/coco/val2017",
        f"{SCRATCH}data/coco/annotations/dev_35_caption.jsonl",
        f"{SCRATCH}data/coco/annotations/train_caption.json",
    shell:
      """
      ln -s {SCRATCH}data/ data/
      """
