localrules: xm3600_images, xm3600_annotations, coco_images, coco_imgz, coco_35_annotations


rule multi30k_trainz:
  output:
    temp("data/multi30k/train.zip")
  shell:
    "wget https://zenodo.org/records/4394718/files/Train.zip -O {output}"

rule multi30k_valz:
  output:
    temp("data/multi30k/val.zip")
  shell:
    "wget https://zenodo.org/records/4394718/files/Val.zip -O {output}"


rule multi30k_caps:
  input:
    "data/multi30k/{split}.zip"
  output:
    directory("data/multi30k/{split}")
  wildcard_constraints:
    split="train|val"
  params:
    split_cap=evaluate("{split}.capitalize()")
  shell:
    """
    cd data/multi30k && unzip {wildcards.split}.zip
    mv {params.split_cap} {wildcards.split}
    """

rule multi30k_img:
  input:
    "flickr30k-images.tar"
  output:
    directory("data/multi30k/images")
  shell:
    """
    tar -xvf {input} -C data/multi30k/
    mv data/multi30k/flickr30k-images data/multi30k/images 
    """

rule multi30k:
  input:
    "data/multi30k/train",
    "data/multi30k/val",
    "data/multi30k/images",
  output:
    temp(touch("data/multi30k/complete"))

rule coco_images:
    input:
      "data/coco/{split}.zip"
    output:
        directory("data/coco/{split}")
    wildcard_constraints:
        split="train2014|val2014|test2014|val2017|train2017"
    shell:
        "cd data/coco && unzip {wildcards.split}.zip"

rule coco_imgz:
  output:
    temp("data/coco/{split}.zip")
  wildcard_constraints:
    split="train2014|val2014|test2014|val2017|train2017"
  shell:
    "wget http://images.cocodataset.org/zips/{wildcards.split}.zip -O {output}"

rule xm3600_images:
    output:
        directory("data/xm3600/images")
    shell:
        """
        mkdir -p data/xm3600/images
        wget https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz
        tar -xvzf images.tgz -C data/xm3600/images
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
        mv coco_mt_{wildcards.split}.jsonl data/coco/annotations/{wildcards.split}_35_caption.jsonl
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

rule coco35:
    input:
      "data/coco/val2017",
      "data/coco/train2017",
      "data/coco/annotations/dev_35_caption.jsonl",
      "data/coco/annotations/train_35_caption.jsonl",
    output:
      temp(touch("data/coco/complete"))

rule xm3600:
  input:
    "data/xm3600/images",
    "data/xm3600/captions.jsonl"
  output:
      temp(touch("data/xm3600/complete"))

  
