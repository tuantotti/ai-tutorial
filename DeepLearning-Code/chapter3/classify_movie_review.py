import numpy as np
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# one-hot encode
def vectorize_sequences(_sequences, _dimension=10000):
    _results = np.zeros((len(_sequences), _dimension))
    for _i, _sequence in enumerate(_sequences):
        _results[_i, _sequence] = 1.
    return _results


x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
