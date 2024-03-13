import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset(data_path:str):
    filelist = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(data_path) for f in filenames]
    X = []
    y = []
    for filename in filelist:
        label = int(filename.split('/')[-2])
        image = np.asarray(Image.open(filename), dtype="int32").transpose()
        X.append(image)
        y.append(label)
    return train_test_split( np.array(X), np.array(y), test_size = 0.5 )