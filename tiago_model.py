import tensorflow as tf
from keras.layers import Dense, Dropout, Input, GRU, LSTM, Embedding
from keras.models import Model
import pandas as pd
import numpy as np
from vocabulary import Vocabulary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import os
from keras.callbacks import EarlyStopping, Callback


run_name = str(10000000000 - round(time()))
os.mkdir(f"runs/{run_name}")

class RecordLossCallback(Callback):
    def __init__(self):
        super(RecordLossCallback, self).__init__()
        with open(f"runs/{run_name}/loss.txt", "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        with open(f"runs/{run_name}/loss.txt", "a") as f:
            f.write(f"{epoch},{logs['loss']},{logs['val_loss']}\n")

def main():
    #load data
    data = pd.read_csv("./data_clean_nosalt_canon_a2d.csv")
    data_y = data["pCHEMBL_norm"]
    data_x = data['SMILES']
    #remove salts
    vocab = Vocabulary("vocab.csv")
    VOCAB_SIZE = vocab.vocab_size
    

    tokenized_smiles = vocab.tokenize(data_x)
    encoded_smiles = vocab.encode(tokenized_smiles)

    data_x = np.array(encoded_smiles)
    #assuming no duplicates in the dataset
    print(data)
    print(data_x.shape)
    print(data_y.shape)

    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

    # Define the model
    input_seq = Input(shape=(train_x.shape[1],))
    encoded = Embedding(VOCAB_SIZE, 128, input_length=train_x.shape[1], mask_zero=True)(input_seq)
    encoded = LSTM(128, return_sequences=True, name="lstm_encoder")(encoded)
    # encoded = Dropout(0.2)(encoded)
    encoded = LSTM(128, return_sequences=False, name="lstm_encoder2")(encoded)
    # encoded = Dropout(0.2)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    # encoded = Dropout(0.2)(encoded)
    encoded = Dense(1)(encoded)

    model = Model(input_seq, encoded)
    model.compile(optimizer='adam', loss='mse') #binary_crossentropy
    model.summary()
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10), RecordLossCallback()]
    model.fit(train_x, train_y, epochs=100, batch_size=16, validation_data=(val_x, val_y), callbacks=callbacks)
    model.save(f"runs/{run_name}/tiago_model.h5")

    preds = model.predict(val_x)
    plt.scatter(preds, val_y)
    plt.show()
    preds = preds.reshape(-1)
    preds = pd.DataFrame(preds, columns=["preds"])
    preds.to_csv(f"runs/{run_name}/val_preds.csv", index=False)


if __name__ == "__main__":

    main()