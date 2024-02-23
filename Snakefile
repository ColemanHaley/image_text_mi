rule coco_images:
    output:
        "data/coco/{split}2014"
    shell:
        """
        wget http://images.cocodataset.org/zips/{wildcards.split}2014.zip
        unzip {wildcards.split}2014.zip
        mv {wildcards.split}2014 data/coco/{wildcards.split}2014/
        """
rule coco_annotations:
    output:
        "data/coco/annotations/train_caption.json"
    shell:
        "wget https://huggingface.co/chaley22/coco_mblip/resolve/main/train_caption.json?download=true -o {output}"

rule coco_val:
    input:
        "data/coco/val2014",
        "data/coco/annotations/train_caption.json",
