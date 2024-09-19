import json
import sys
from pathlib import Path

from iso639 import Lang
from PIL import Image
from torch.utils.data import Dataset
from tqdm.rich import tqdm


class XM3600Dataset(Dataset):
    def __init__(self, data_dir, lang, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.lang = lang
        with open(Path(self.data_dir) / "xm3600" / "captions.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            for c in cap[self.lang]["caption/tokenized"]:
                data.append({"caption": c, "image": cap["image/key"]})
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        path = self.data_dir / "xm3600" / "images" / f"{caption['image']}.jpg"
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["caption"], caption["image"]


class XM3600TextDataset(Dataset):
    def __init__(self, data_dir, lang="all", transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.lang = lang
        with open(Path(self.data_dir) / "xm3600" / "captions.jsonl", "r") as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            for key in cap.keys():
                if (self.lang == "all" or key in self.lang) and len(key) == 2:
                    for c in cap[key]["caption/tokenized"]:
                        data.append({"caption": c, "lang": key})
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        return f'caption {caption["lang"]}\n{caption["caption"]}'


class COCO35TextDataset(Dataset):
    def __init__(self, data_dir, split, lang="all", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["dev", "train", "dev_sub"]
        self.lang = lang

        self.transform = transform
        with open(
            Path(self.data_dir) / "annotations" / f"{self.split}_35_caption.jsonl", "r"
        ) as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        ens = set([])
        for cap in tqdm(captions):
            if self.lang == "all" or cap["trg_lang"] in self.lang:
                data.append(cap)
            if "en" in self.lang or self.lang == "all":
                if cap["src_lang"] == "en" and cap["caption_tokenized"] not in ens:
                    ens.add(cap["caption_tokenized"])
                    data.append(
                        {
                            "trg_lang": "en",
                            "translation_tokenized": cap["caption_tokenized"],
                            "image_id": cap["image_id"],
                        }
                    )
        return data  # should be `data` if you revive this method!

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        return f'caption {caption["trg_lang"]}\n{caption["translation_tokenized"]}'


class COCO35Dataset(Dataset):
    def __init__(self, data_dir, split, lang, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["dev"]
        self.transform = transform
        self.lang = lang
        with open(
            Path(self.data_dir) / "annotations" / "dev_35_caption.jsonl", "r"
        ) as f:
            captions = [json.loads(jline) for jline in f.readlines()]
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            img_id = int(cap["image_id"].split("_")[0])
            path = self.data_dir / "val2017" / f"{img_id:012d}.jpg"
            path2 = self.data_dir / "train2017" / f"{img_id:012d}.jpg"
            if (
                cap["trg_lang"] == self.lang
                or self.lang == "en"
                and cap["trg_lang"] == "fr"
            ):
                if path.is_file():
                    cap["image_path"] = path
                    data.append(cap)
                elif path2.is_file():
                    cap["image_path"] = path2
                    data.append(cap)
                else:
                    print(f"Image not found: {path}", file=sys.stderr)
            if self.lang == "en" and cap["trg_lang"] == "fr":
                cap["translation_tokenized"] = cap["caption_tokenized"]
                cap["trg_lang"] = "en"
        return data

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"].split("_")[0])
        # img_id = int(caption["image_id"].split("_")[0])
        img = Image.open(caption["image_path"])
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["translation_tokenized"], img_id


class Multi30kDataset(Dataset):
    def __init__(self, data_dir, split, lang, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["train", "val"]
        self.transform = transform
        self.lang = lang
        try:
            with open(self.data_dir / split / f"{Lang(self.lang).name}.txt", "r") as f:
                captions = f.readlines()
        except UnicodeDecodeError:
            with open(
                self.data_dir / split / f"{Lang(self.lang).name}.txt",
                "r",
                encoding="cp1252",
            ) as f:
                captions = f.readlines()
        with open(self.data_dir / split / "IDs.txt") as f:
            ids = f.read().splitlines()
        self.captions = [
            {"caption": cap, "image_id": id} for cap, id in zip(captions, ids)
        ]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img = Image.open(self.data_dir / "images" / caption["image_id"])
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        if self.transform is not None:
            img = self.transform(images=img)
        return img, caption["caption"].strip(), caption["image_id"]


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
