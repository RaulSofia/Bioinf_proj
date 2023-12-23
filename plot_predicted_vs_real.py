from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model

from vocabulary import Vocabulary


def main():
    predictor = load_model("encoder_predictor.h5")
    predictor.summary()
    #load data
    data = pd.read_csv("./data_clean_nosalt_canon_a2d.csv")
    data_y = data["pCHEMBL_norm"]
    data_x = data['SMILES']
    vocab = Vocabulary("vocab.csv")
    VOCAB_SIZE = vocab.vocab_size
    tokenized_smiles = vocab.tokenize(data_x)
    encoded_smiles = vocab.encode(tokenized_smiles)
    data_x = np.array(encoded_smiles)
    preds = predictor.predict(data_x)

    preds = preds.reshape(-1)
    print(preds.shape)
    print(data_y.shape)
    
    plt.scatter(preds, data_y)

    plt.show()


if __name__ == "__main__":
    main()