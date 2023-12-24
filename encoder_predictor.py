import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
import pandas as pd
import numpy as np
from vocabulary import Vocabulary
from rdkit import Chem
from rdkit.Chem import SaltRemover
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import os
from keras.callbacks import EarlyStopping, Callback

# tf.config.set_visible_devices([], 'GPU')

params = {
    "model_name": "encoder_32.h5",
    "encoder_freezed": False,
    "start_freezed": False,
    "batch_size": 16,
    "epochs": 100,
    "patience": 10,
    "optimizer": 'adam',
    "loss": "mse",
    "reset_weights": False,
    "dropout": 0.4,
    "dataset": "data_clean_nosalt_canon_a2d.csv"
}

# assert params["encoder_freezed"] or params["start_freezed"], "Must freeze the encoder or start freezed"

class RecordLossCallback(Callback):
    def __init__(self):
        super(RecordLossCallback, self).__init__()
        with open(f"runs/{run_name}/loss.txt", "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    def on_epoch_end(self, epoch, logs=None):
        with open(f"runs/{run_name}/loss.txt", "a") as f:
            f.write(f"{epoch},{logs['loss']},{logs['val_loss']}\n")

def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)

if __name__ == "__main__":
    run_name = str(10000000000 - round(time()))
    os.mkdir(f"runs/{run_name}")
    encoder = load_model(params["model_name"])
    #freeze the encoder
    if params["encoder_freezed"] or params["start_freezed"]:
        encoder.trainable = False
    else:
        encoder.trainable = True
    encoder.summary()

    # Define the model
    input_seq = Input(shape=(encoder.output.shape[1],))
    encoded = Dropout(params["dropout"])(input_seq)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dropout(params["dropout"])(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dropout(params["dropout"])(encoded)
    encoded = Dense(1, activation='sigmoid')(encoded)

    #load data
    data = pd.read_csv(f"./{params['dataset']}")
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

    # Encoder based predictor model
    predictor = Model(input_seq, encoded)
    predictor.summary()
    # predictor.compile(optimizer=params['optimizer'], loss=params['loss']) #binary_crossentropy
    # train_x = encoder.predict(train_x)
    # val_x = encoder.predict(val_x)
    print("X:", data_x)
    print("Y:", data_y)
    callbacks = [EarlyStopping(monitor='val_loss', patience=params["patience"], restore_best_weights=True), RecordLossCallback()]
    
    # predictor.fit(train_x, train_y, epochs=100, batch_size=16, shuffle=True, validation_data=(val_x, val_y), callbacks=callbacks)

    

    full_model = Model(encoder.input, predictor(encoder.output))
    full_model.summary()
    if params["reset_weights"]:
        print("!!!!!!!WARNING: Resetting weights!!!!!!!!")
        model_cloned = tf.keras.models.clone_model(full_model)


    full_model.compile(optimizer=params['optimizer'], loss='mse') #binary_crossentropy
    full_model.fit(train_x, train_y, epochs=params['epochs'], batch_size=params['batch_size'], shuffle=True, validation_data=(val_x, val_y), callbacks=callbacks)
    
    if params["start_freezed"]:
        for layer in full_model.layers:
            layer.trainable = True
        full_model.compile(optimizer=params['optimizer'], loss='mse')
        full_model.summary()
        full_model.fit(train_x, train_y, epochs=params['epochs'], batch_size=params['batch_size'], shuffle=True, validation_data=(val_x, val_y), callbacks=callbacks)
    # full_model.save(f"runs/{run_name}/full_model.h5")

    val_preds = full_model.predict(val_x)
    val_preds = val_preds.reshape(-1)
    print(val_preds.shape, val_y.shape) 
    preds_data = pd.DataFrame({"preds": val_preds, "y": val_y})
    preds_data.to_csv(f"runs/{run_name}/val_preds.csv")
    params_df = pd.DataFrame(params, index=[0])
    params_df.to_csv(f"runs/{run_name}/params.csv")
    plt.scatter(val_preds, val_y)
    plt.show()

    