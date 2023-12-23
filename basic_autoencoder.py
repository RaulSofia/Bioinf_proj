#TODO: work with smiles
#TODO: smiles to tokens
#TODO: tokens to embeddings
#TODO: adjust input to accomodate embeddings
#TODO: mirror the encoder and decoder
#TODO: change loss function to binary_crossentropy
#TODO: find large dataset to preptrain encoder and decoder
#TODO: create model to predict binding affinity (1, 2, 3 layers)
#TODO: train this last model on a dataset composed of encoder(tiago's smiles) -> binding affinity



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, RepeatVector, LSTM, TimeDistributed, Flatten, Embedding, Dropout
from keras.models import Model
from vocabulary import Vocabulary
# Early Stopping
from keras.callbacks import EarlyStopping

# mirrored_strategy = tf.distribute.MirroredStrategy()
tf.config.set_visible_devices([], 'GPU')

ENCODING_DIM = 128
EMBEDDING_DIM = 128
MAX_LEN = 100
VOCAB_SIZE = 20

# configproto = tf.compat.v1.ConfigProto() 
# configproto.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=configproto) 
# tf.compat.v1.keras.backend.set_session(sess)

def main():
    # Generate example data
    vocab = Vocabulary("vocab.csv")
    VOCAB_SIZE = vocab.vocab_size
    dataset = vocab.load_dataset("250k_rndm_zinc_drugs_clean_3.csv", 500)
    print(dataset)

    tokenized = vocab.tokenize(dataset)
    encoded = vocab.encode(tokenized)
    one_hot = vocab.one_hot_encoder(tokenized)

    data = np.array(encoded)
    data_y = np.array(one_hot)
    print(data)

    # with mirrored_strategy.scope():
        # Define the model
    input_seq = Input(shape=(MAX_LEN,))
    encoded = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True)(input_seq)
    encoded = LSTM(512, return_sequences=True, name="lstm_encoder")(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(256, return_sequences=True, name="lstm_encoder2")(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = TimeDistributed(Dense(ENCODING_DIM))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(ENCODING_DIM, name="encoder_output")(encoded)

    # Decoder layers
    decoded = RepeatVector(MAX_LEN)(encoded)
    decoded = LSTM(256, return_sequences=True, name="lstm_decoder")(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = LSTM(512, return_sequences=True, name="lstm_decoder2")(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoded)

    # Autoencoder model
    autoencoder = Model(input_seq, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy') #binary_crossentropy

    print(tf.config.list_physical_devices('GPU'))

    # Early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Train the model
    history = autoencoder.fit(data, data_y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, callbacks=[es])

    encoder = Model(inputs=input_seq, outputs=autoencoder.get_layer('encoder_output').output)
    # Encode and decode some data
    encoded_data = encoder.predict(data)
    print(encoded_data.shape)

    encoder.save("encoder_weights.h5")

if __name__ == "__main__":
    main()