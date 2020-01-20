
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from utils.constants import NO_TRAIN, NO_TEST, NO_POINTS


class LatentStateModel:
    """"""
    normalizer = None
    regressor = None
    x_train, y_train = None, None
    x_test, y_predicted = None, None

    def __init__(self, df_row: np.array):
        """"""
        self.x_train, self.y_train = [], []
        self.x_test, self.y_predicted = [], []

        self.df_row = df_row
        self.train = df_row[0:NO_TRAIN].values
        self.test = df_row[NO_TRAIN:NO_POINTS].values
        # Init normalization method: for all x_i = X_i - min(x) / max(x)-min(x)
        self.normalizer = MinMaxScaler(feature_range=(0, 1))
        self._preprocessing()
        self._build_recurrent_network()

    def _preprocessing(self):
        """Preprocessing training & test data."""
        _l = 30  # Number of PRNG sequences to use to train each step of the LSTM.

        # Preprocessing Training data for each LSTM training step.
        train_normd = self.normalizer.fit_transform(self.train.reshape(-1, 1))
        for i in range(_l, NO_TRAIN):
            self.x_train.append(train_normd[i - _l:i, 0])
            self.y_train.append(train_normd[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        # Preprocessing Test data for each LSTM prediction step.
        # Note we normalize a combination of some of the training data along with the test data for ~consistent scaling.
        dataset_total = self.df_row[0: NO_POINTS]  # Drops the 'seed' column
        inputs = dataset_total[NO_POINTS - NO_TEST - _l:].values
        test_normd = self.normalizer.fit_transform(inputs.reshape(-1, 1))
        for i in range(_l, _l+NO_TEST):
            self.x_test.append(test_normd[i - _l:i, 0])
        self.x_test = np.array(self.x_test)
        self. x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))
        
    def _build_recurrent_network(self):
        """Builds the NN architecture to from a LSTM."""
        self.regressor = Sequential()
        # Layers 1 to 4 with Dropout regularisation.
        self.regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))
        self.regressor.add(LSTM(units=50))
        self.regressor.add(Dropout(0.2))
        # Output layer.
        self.regressor.add(Dense(units=1))
        # Compile LSTM.
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    def train_recurrent_network(self):
        """"""
        # Fitting the RNN to the Training set
        self.regressor.fit(self.x_train, self.y_train, epochs=100, batch_size=32, verbose=0)  # verbose=1 progress bar
        
    def predict_values(self):
        """"""
        y_predicted = self.regressor.predict(self.x_test)
        self.y_predicted = self.normalizer.inverse_transform(y_predicted).astype(int)

    def rmse(self):
        """"""
        return math.sqrt(mean_squared_error(self.test, self.y_predicted))
