import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Load data
data = np.loadtxt('data.csv', delimiter=',')

# Define the input layer
input_layer = Input(shape=(data.shape[1],))

# Define the encoder layers
def build_encoder(hp):
    encoded = Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=16), activation='relu')(input_layer)
    encoded = Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu')(encoded)
    encoded = Dense(units=hp.Int('units_3', min_value=8, max_value=32, step=8), activation='relu')(encoded)
    return encoded

# Define the decoder layers
def build_decoder(hp, encoded):
    decoded = Dense(units=hp.Int('units_4', min_value=8, max_value=32, step=8), activation='relu')(encoded)
    decoded = Dense(units=hp.Int('units_5', min_value=16, max_value=64, step=16), activation='relu')(decoded)
    decoded = Dense(units=hp.Int('units_6', min_value=32, max_value=128, step=16), activation=None)(decoded)
    return decoded

# Define the autoencoder model
def build_autoencoder(hp):
    encoded = build_encoder(hp)
    decoded = build_decoder(hp, encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Define the tuner
tuner = RandomSearch(build_autoencoder, objective='val_loss', max_trials=10, executions_per_trial=2)

# Perform hyperparameter tuning
tuner.search(x=data, y=data, epochs=100, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping('val_loss', patience=3)])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Use the encoder to transform the data
encoder = Model(inputs=input_layer, outputs=best_model.layers[2].output)
encoded_data = encoder.predict(data)

