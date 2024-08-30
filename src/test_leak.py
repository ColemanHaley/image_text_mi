from dataset import COCO35TextDataset, COCO35Dataset

LANG = "th"
train_dataset = COCO35TextDataset(
    "data/coco",
    "train",
    lang=LANG,
)

test_dataset = COCO35Dataset(
    "data/coco",
    "dev",
    lang=LANG,
)
print(len(test_dataset.captions))


caps = set(x["translation_tokenized"] for x in train_dataset.captions)
test_caps = set(x["translation_tokenized"] for x in test_dataset.captions)
print(train_dataset.captions[0])
# check intersection
print(f"Train captions in test: {len(caps.intersection(test_caps))}")

img_ids = set(x["image_id"] for x in train_dataset.captions)
test_img_ids = set(x["image_id"] for x in test_dataset.captions)
print(f"Train images in test: {len(img_ids.intersection(test_img_ids))}")
print(list(caps.intersection(test_caps))[:5])
print(
    "ein Mann , der am Strand einen Drachen steigen lässt ."
    in caps.intersection(test_caps)
)
