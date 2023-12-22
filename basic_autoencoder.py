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

EMBEDDING_DIM = 128

def main():
    # Generate example data
    data = np.random.random((1000, 10, 1))  # Replace this with your own data
    print(data)

    # Define the model
    input_seq = Input(shape=(10, 1))
    # encoded = Embedding(10, EMBEDDING_DIM, input_length=10)(input_seq)
    encoded = LSTM(128, return_sequences=True, name="lstm_encoder")(input_seq)
    encoded = TimeDistributed(Dense(EMBEDDING_DIM))(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(EMBEDDING_DIM, name="encoder_output")(encoded)

    decoded = RepeatVector(10)(encoded)

    # Decoder layers
    decoded = LSTM(128, return_sequences=True, name="lstm_decoder")(decoded)

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