import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, RepeatVector, LSTM, TimeDistributed, Flatten
from tensorflow.keras.models import Model


def main():
    # Generate example data
    data = np.random.random((1000, 10, 1))  # Replace this with your own data
    print(data)

    # Define the model
    input_seq = Input(shape=(10, 1))
    encoded = LSTM(128, return_sequences=True, name="lstm_encoder")(input_seq)
    encoded = TimeDistributed(Dense(1))(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(128, name="encoder_output")(encoded)

    decoded = RepeatVector(10)(encoded)

    # Decoder layers
    decoded = LSTM(128, return_sequences=True, name="lstm_decoder")(decoded)

    # Autoencoder model
    autoencoder = Model(input_seq, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    

    # Train the model
    autoencoder.fit(data, data, epochs=1, batch_size=32, shuffle=True, validation_split=0.2)

    encoder = Model(inputs=input_seq, outputs=autoencoder.get_layer('encoder_output').output)
    # Encode and decode some data
    encoded_data = encoder.predict(data)
    print(encoded_data.shape)

if __name__ == "__main__":
    main()