import json
import sys
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from tqdm.rich import tqdm

class XM3600Dataset(Dataset):
    def __init__(self, data_dir, lang, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.lang=lang
        with open(Path(self.data_dir) / "xm3600" / "captions.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            for c in cap[self.lang]['caption']:
                data.append({'caption': c, 'image': cap['image/key']})
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        path = (
            self.data_dir
            / "xm3600" / f"{caption['image']}.jpg"
        )
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["caption"], caption['image']


class COCO35TextDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["dev", "train"]
        
        self.transform = transform
        with open(Path(self.data_dir) / "annotations" / f"{self.split}_35_caption.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)
        path = self.data_dir / "blank.jpg"
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        self.img = img

    def _get_split(self, captions):
        data = []
        # for cap in tqdm(captions):
        #     img_id = int(cap["image_id"].split("_")[0])
        #     path = (
        #         self.data_dir
        #         / f"val2014" / "val2014"
        #         / f"COCO_val2014_{img_id:012d}.jpg"
        #     )

        #     if self.split == "train" and not path.is_file():
        #         data.append(cap)
        #     elif self.split == "dev" and path.is_file():
        #         data.append(cap)
        #     else:
        #         print(f"Image not found: {path}", file=sys.stderr)

        return captions # should be `data` if you revive this method!

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        return f'caption {caption["trg_lang"]}\n{caption["translation_tokenized"]}'


class COCO35LMDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["dev", "train"]
        
        self.transform = transform
        with open(Path(self.data_dir) / "annotations" / f"{self.split}_35_caption.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)
        path = self.data_dir / "blank.jpg"
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        self.img = img

    def _get_split(self, captions):
        data = []
        # for cap in tqdm(captions):
        #     img_id = int(cap["image_id"].split("_")[0])
        #     path = (
        #         self.data_dir
        #         / f"val2014" / "val2014"
        #         / f"COCO_val2014_{img_id:012d}.jpg"
        #     )

        #     if self.split == "train" and not path.is_file():
        #         data.append(cap)
        #     elif self.split == "dev" and path.is_file():
        #         data.append(cap)
        #     else:
        #         print(f"Image not found: {path}", file=sys.stderr)

        return captions # should be `data` if you revive this method!

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"].split("_")[0])
        return self.img, caption["translation_tokenized"], caption['trg_lang']


class COCO35Dataset(Dataset):
    def __init__(self, data_dir, split, lang, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["dev"]
        self.transform = transform
        self.lang=lang
        with open(Path(self.data_dir) / "annotations" / "dev_35_caption.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            img_id = int(cap["image_id"].split("_")[0])
            path = (
                self.data_dir
                / f"val2014" / "val2014"
                / f"COCO_val2014_{img_id:012d}.jpg"
            )
            if cap["trg_lang"] == self.lang:
                if path.is_file():
                    data.append(cap)
                else:
                    print(f"Image not found: {path}", file=sys.stderr)

        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"].split("_")[0])
        path = (
            self.data_dir
            / f"val2014" / "val2014"
            / f"COCO_val2014_{img_id:012d}.jpg"
        )
        #img_id = int(caption["image_id"].split("_")[0])
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["translation_tokenized"], img_id

class COCODataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["train", "val"]
        self.transform = transform
        with open(Path(self.data_dir) / "annotations" / "train_caption.json", "r") as f:
            captions = json.load(f)
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            path = (
                self.data_dir
                / f"{self.split}2014"
                / f"COCO_{self.split}2014_{int(cap['image_id']):012d}.jpg"
            )
            if path.is_file():
                data.append(cap)
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"])
        path = (
            self.data_dir
            / f"{self.split}2014"
            / f"COCO_{self.split}2014_{img_id:012d}.jpg"
        )
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["caption"], img_id

