import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# build the network
def build_model():
    _model = models.Sequential()
    _model.add(layers.Dense(64, activation='relu',
                            input_shape=(train_data.shape[1],)))
    _model.add(layers.Dense(64, activation='relu'))
    _model.add(layers.Dense(1))
    _model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return _model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    validation_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    validation_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
                                     axis=0)

    model = build_model()
    history = model.fit(partial_data, partial_targets, validation_data=(validation_data, validation_targets),
                        epochs=num_epochs, batch_size=1,
                        verbose=0,
                        )  # verbose=0 is silent mode for logging
    mae_history = history.history['val_mae']
    all_scores.append(mae_history)

print(all_scores)
