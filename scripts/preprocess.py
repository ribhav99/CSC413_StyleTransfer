import os
from PIL import Image
from tqdm import tqdm

root_dir = '../data/trainMNIST/'
for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in tqdm(files):
        if name.endswith(".jpg") or name.endswith(".png"):
            im = Image.open(root + name)
            # imResize = im.resize((256, 256), Image.ANTIALIAS)
            # os.remove(root_dir + name)
            # imResize.save(root_dir + name + ".jpg", 'JPEG', quality=90)
            im += 1
            im *= 127.5
            im.save('../try/' + name + ".jpg", 'JPEG', quality=90)
