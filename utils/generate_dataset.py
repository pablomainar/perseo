from trdg.generators import GeneratorFromStrings
import numpy as np
import os
import shutil
import pandas as pd
import pickle as pkl
from multiprocess import Process, Manager


def preprocess_text(text, min_words=None, max_words=None):
    sentences = text.split(".")
    idx_remove = []
    for idx, sentence in enumerate(sentences):
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("thumb|left|", " ")
        sentence = sentence.replace("thumb|right|", " ")
        sentence = sentence.replace("thumb|", " ")
        sentence = sentence.replace(";", " ")
        sentence = sentence.split()
        sentence = " ".join(sentence)
        if min_words:
            if len(sentence.split()) < min_words:
                idx_remove.append(idx)
        if max_words:
            if len(sentence.split()) > max_words:
                sentence = " ".join(sentence.split()[:max_words])
        sentences[idx] = sentence
    sentences = np.array(sentences)
    sentences = np.delete(sentences, idx_remove)
    return sentences

def store_data(process_id, generator, labels_dict, sentences_per_process):
        labels = {}
        labels["path"] = []
        labels["text"] = []
        df = pd.DataFrame(labels)
        for count, (image, text) in enumerate(generator):
            if image is None:
                print("Image is None: ", count, text)
                continue
            name = process_id*sentences_per_process + count
            sample_path = os.path.join(datapath, set_, str(name)+".jpg")
            image.save(sample_path)
            labels_dict[name] = text
            labels["path"].append(str(name)+".jpg")
            labels["text"].append(text)
            if count % 1000 == 0:
                df = pd.concat([df, pd.DataFrame(labels)])
                df.to_csv(os.path.join(datapath, set_+".csv"), index=False)
                labels = {}
                labels["path"] = []
                labels["text"] = []
                

if __name__ == "__main__":
    set_ = "test"

    datapath = "../data_handwritten_test"

    print("Loading dataset...")
    list_ds = pkl.load(open("../wikipedia_test.pkl", "rb"))
    
    print("Starting...")
    sentences = []
    print(len(list_ds))


    for count, entry in enumerate(list_ds):
        text = entry["text"].numpy().decode("utf-8")
        sentences_entry = preprocess_text(text, min_words=3, max_words=25)
        sentences.extend(sentences_entry)

    nb_processes = 1
    sentences_per_process = len(sentences) // nb_processes
    sentences = [sentences[i:i + sentences_per_process] for i in range(0, len(sentences), sentences_per_process)]
    sentences = sentences[:nb_processes]

    fonts = [os.path.abspath(os.path.join("fonts", "ttfs", p)) for p in os.listdir(os.path.join("fonts", "ttfs"))]

    generators = [GeneratorFromStrings(sentences[i], language="es", count=sentences_per_process, fonts=fonts) for i in range(nb_processes)]
    manager = Manager()
    labels_dict = manager.dict()

    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.mkdir(datapath)
    if not os.path.exists(os.path.join(datapath, set_)):
        os.mkdir(os.path.join(datapath, set_))

    processes = [Process(target=store_data, args=(i, generators[i], labels_dict, sentences_per_process)) for i in range(nb_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
