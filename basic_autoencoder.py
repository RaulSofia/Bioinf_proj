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
from tensorflow.keras.layers import Input, Dense, RepeatVector, LSTM, TimeDistributed, Flatten, Embedding
from tensorflow.keras.models import Model

tf.config.set_visible_devices([], 'GPU')

ENCODING_DIM = 128
EMBEDDING_DIM = 128
MAX_LEN = 100
VOCAB_SIZE = 20

def main():
    # Generate example data
    data = np.random.randint(0, VOCAB_SIZE-1, size=(1000, MAX_LEN))  # Replace this with your own data
    print(data)

    # Define the model
    input_seq = Input(shape=(MAX_LEN,))
    print("input_seq:", input_seq.shape)
    encoded = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True)(input_seq)
    print("encoded_embedding:", encoded.shape)
    encoded = LSTM(128, return_sequences=True, name="lstm_encoder")(encoded)
    print("encoded_lstm:", encoded.shape)
    encoded = TimeDistributed(Dense(ENCODING_DIM))(encoded)
    print("encoded_time_distributed:", encoded.shape)
    encoded = Flatten()(encoded)
    print("encoded_flatten:", encoded.shape)
    encoded = Dense(ENCODING_DIM, name="encoder_output")(encoded)
    print("encoded_dense:", encoded.shape)

    decoded = RepeatVector(MAX_LEN)(encoded)

    # Decoder layers
    decoded = LSTM(128, return_sequences=True, name="lstm_decoder")(decoded)
    decoded = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoded)

    # Autoencoder model
    autoencoder = Model(input_seq, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error') #binary_crossentropy
    

    # Train the model
    autoencoder.fit(data, data, epochs=1, batch_size=32, shuffle=True, validation_split=0.2)

    encoder = Model(inputs=input_seq, outputs=autoencoder.get_layer('encoder_output').output)
    # Encode and decode some data
    encoded_data = encoder.predict(data)
    print(encoded_data.shape)

if __name__ == "__main__":
    main()