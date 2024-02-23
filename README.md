This repo contains code for captioning images and comparing the log probabilities of the correct captions to their probabilities under a tuned language model. 

First, install the requirements:
```
pip install -r requirements.txt
```

To run the code on the two test images with a few test captions:
```
python src/caption.py
```

To run the code on the COCO 2014 validation set, first download the dataset
```
snakemake --cores all coco_val
```
then run the following command
```
python src/caption.py dataset=coco
```

You can mess with hyperparameters using the config.yaml or Hydra command-line syntax.
