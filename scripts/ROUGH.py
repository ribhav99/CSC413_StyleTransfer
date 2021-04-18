import imageio
import os

direc = "../data/testZebra/"
for root, dirs, files in os.walk(direc, topdown=False):
    for name in files:
        if name.endswith(".jpg") or name.endswith(".png"):
            if len(imageio.imread(direc + name).shape) != 3:
                print(name)
                os.remove(direc + name)
