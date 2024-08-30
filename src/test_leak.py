from dataset import COCO35TextDataset, COCO35Dataset

LANG = 'th'
train_dataset = COCO35TextDataset(
       'data/coco',
       'train',
       lang=LANG,
    )

test_dataset = COCO35Dataset(
         'data/coco',
         'dev',
         lang=LANG,
    )
# test_dataset = COCO35TextDataset(
#        'data/coco',
#        'dev',
#        lang=LANG,
#     )
print(len(test_dataset.captions))


caps = set(x['translation_tokenized'] for x in train_dataset.captions)
test_caps = set(x['translation_tokenized'] for x in test_dataset.captions)
print(train_dataset.captions[0])
# check intersection
print(len(caps.intersection(test_caps)))
# check other intersection
print(len(test_caps.intersection(caps)))
print(list(caps.intersection(test_caps))[:5])
print("ein Mann , der am Strand einen Drachen steigen lässt ." in caps.intersection(test_caps))


