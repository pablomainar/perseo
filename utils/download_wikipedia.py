from trdg.generators import GeneratorFromStrings
import tensorflow_datasets as tfds
import numpy as np
import os
import shutil
import pandas as pd
import pickle as pkl

ds = tfds.load('wikipedia/20230601.es')
list_ds = list(ds["train"])
with open("wikipedia.pkl", "wb") as f:
    pkl.dump(list_ds, f)