import tensorflow_datasets as tfds
import pickle as pkl

ds = tfds.load('wikipedia/20230601.es')
list_ds = list(ds["train"])
with open("wikipedia.pkl", "wb") as f:
    pkl.dump(list_ds, f)
