from dataset import COCO35TextDataset, COCO35Dataset

train_dataset = COCO35TextDataset(
       'data/coco',
       'train',
       lang='de',
    )

test_dataset = COCO35Dataset(
         'data/coco',
         'dev',
         'de'
    )

caps = set(x['translation_tokenized'] for x in train_dataset.captions)
test_caps = set(x['translation_tokenized'] for x in test_dataset.captions)
# check intersection
print(len(caps.intersection(test_caps)))
# check other intersection
print(len(test_caps.intersection(caps)))

