from PIL import Image


def make_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert(mode="RGB")
    return img
