import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import KernelPCA
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def build_sequential_mlp_reg(n_feature=64,
                             n_output=1,
                             activation='relu',
                             hidden_size=40,
                             hidden_size_2=None,
                             kernel_initializer='glorot_uniform',
                             dropout=0.25,
                             dropout_2=0.,
                             learning_rate=0.001,
                             beta_1=0.9,
                             beta_2=0.999,
                             batch_normalisation=False,
                             last_activation='linear',
                             loss="mse"):
    """build a sequential mlp regressor nn"""

    x = Sequential()
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation='relu',
                input_dim=n_feature))
    x.add(Dropout(dropout))
    if batch_normalisation and activation not in ['linear']:
        x.add(BatchNormalization())
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation=activation))
    x.add(Dropout(dropout))
    if hidden_size_2 is not None:
        if batch_normalisation and activation not in ['linear']:
            x.add(BatchNormalization())
        x.add(Dense(hidden_size_2,
                    kernel_initializer=kernel_initializer,
                    activation=activation))
        x.add(Dropout(dropout_2))

    x.add(Dense(units=n_output,
                kernel_initializer=kernel_initializer,
                activation=last_activation))

    optimizer = keras.optimizers.Adam(lr=learning_rate,
                                      beta_1=beta_1,
                                      beta_2=beta_2)
    x.compile(loss=loss,
              optimizer=optimizer,
              metrics=['mse'])
    return x


def build_sequential_cnn_reg(input_shape,
                             kernel_size=(10, 10),
                             filters=32,
                             activation='relu',
                             n_output=1,
                             hidden_size=500,
                             use_bias=False,
                             learning_rate=0.01,
                             bias_initializer='random_uniform',
                             loss=keras.losses.mean_squared_error
                             ):
    model = Sequential()
    model.add(Conv2D(filters,
                     kernel_size=kernel_size,
                     activation=activation,
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(hidden_size, use_bias=True,
                    bias_initializer=bias_initializer,
                    activation=activation))
    model.add(Dense(n_output,
                    use_bias=use_bias,
                    bias_initializer=bias_initializer))
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adadelta(lr=learning_rate, rho=0.95, decay=0.0))
                  #optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, decay=0.0))
    return model


def monte_carlo_sampling(keras_model, X_test,  MC_path=100):
    """take a fitted keras model implementing dropout
    and generate forward path into the network resulting
     in different prediction for X_test"""
    MC_output = K.function([keras_model.layers[0].input, K.learning_phase()],
                           [keras_model.layers[-1].output])
    learning_phase = True  # use dropout at test time

    MC_samples = [MC_output([X_test, learning_phase])[0] for _ in range(MC_path)]
    MC_samples = np.array(MC_samples)

    return MC_samples


def compute_uncertainty(MC_samples):
    """compute MC mean, var and confidence"""
    mean = np.mean(MC_samples, axis=0)
    std = np.std(MC_samples, axis=0)
    return mean, std, mean - 1.96 * std, mean + 1.96 * std


def q_loss(q, y, f):
    """quantile loss function"""
    e = (y - f)
    return keras.backend.mean(keras.backend.maximum(q * e, (q - 1) * e),
                              axis=-1)


class MLPRegressor(BaseEstimator, RegressorMixin):

    """Class implementing a regressor (which support quantile)
     using sequential model.

    Args
    ----------
    scaler: Callable
        scaler used to normalize or preprocess input
    num_class: int
        input dimension of the network
    n_feature: int
        nb of feature
    model: Keras.model
        Regression Keras model
    epachs: int
        number of epochs
    batch_size: int
        number of element in each batch
    alpha: float (0<=_<=1)
        alpha for quantile regression
    verbose: int
        verbosity of self.model.fit() method

    """
    def __init__(self, epochs, batch_size, scaler=StandardScaler(),
                 loss='mean_squared_error', n_output=1,
                 n_feature=10, alpha=None, verbose=0, **kwargs):
        self._estimator_type = 'regressor'
        self.scaler = scaler
        if alpha is not None:
            self.loss = lambda y, f: q_loss(alpha, y, f)
        else:
            self.loss = loss
        self.n_feature = n_feature
        self.n_output = n_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.nn_params = kwargs
        self.model = self._create_model()

    def _create_model(self):
        keras.backend.clear_session()
        return build_sequential_mlp_reg(n_feature=self.n_feature,
                                        n_output=self.n_output,
                                        loss=self.loss,
                                        **self.nn_params)

    def fit(self, X, y, sample_weight=None, reset_model=True, **kwargs):
        """
        Fit the model.

        Args
        ----------
        X: array-like
            Feature space
        y: array-like
            target space
        sample_weight: array-like, default None
            sample weights
        reset_model: bool, default True
            reset model before fitting, needed for parameters search or cv

        kwargs: dict
            arguments passed to self.model.fit()
        """
        X = self.scaler.fit_transform(X)
        if reset_model:
            self.model = self._create_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=self.verbose, sample_weight=sample_weight, **kwargs)
        return self

    def predict(self, X, nb_path=None, sampling=False, **kwargs):
        """
        Prediction.

        Args
        ----------
        X: array-like
            Feature space
        nb_path: int or None
            number of forward path in network used for prediction
        sampling: bool, default None
            weather to return an array of shape (nb_path, n_sample, 1)
            containing all prediction paths
        kwargs: dict
            arguments passed to self.model.predict()
        """
        X = self.scaler.fit_transform(X)
        if nb_path is not None:
            mc_samples = monte_carlo_sampling(self.model,
                                              X_test=X,
                                              MC_path=nb_path)
            if sampling:
                return mc_samples
            else:
                return compute_uncertainty(mc_samples)
        else:
            pred = self.model.predict(X, **kwargs)
            return pred

    def set_params(self, **params):
        self.nn_params = params
        self.model = self._create_model()
        return self


class MLPRegressorPCA(MLPRegressor):

    """Class implementing a classifier using sequential mlp model
     with PCA preprocessing.

    Args
    ----------
    n_feature: int
        nb of feature to keep for variance explaination
    kwargs: dict
        any parameters of MLPRegressor
    """
    def __init__(self, n_feature=10, **kwargs):
        self.pca = PCA(n_components=n_feature)
        super(MLPRegressorPCA, self).__init__(n_feature=n_feature, **kwargs)

    def fit(self, X, y, **kwargs):
        """
        Fit the model.

        Args
        ----------
        X: array-like
            Feature space
        y: array-like
            target space
        kwargs: dict
            arguments passed to super().fit()
        """
        X = self.scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        super().fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Prediction.

        Args
        ----------
        X: array-like
            Feature space
        kwargs: dict
            arguments passed to super().predict()
        """
        X = self.scaler.fit_transform(X)
        X = self.pca.transform(X)
        return super().predict(X, **kwargs)

    def set_params(self, **params):
        if 'n_feature' in params.keys():
            self.n_feature = params['n_feature']
            del params['n_feature']
        self.pca = PCA(n_components=self.n_features)
        super().set_params(**params)


class CNNRegressor(BaseEstimator, RegressorMixin):

    """Class implementing a regressor using CNN structure.

    Args
    ----------
    input_shape: tuple
        input shape
    n_output: int
        number of output neurons
    epochs: int
        number of epochs
    batch_size: int
        number of element in each batch
    verbose: int
        verbosity of self.model.fit() method
    loss: str or callable
        loss for the sequential model
    model: Keras.Sequential
        a sequential Keras model

    """
    def __init__(self, epochs, batch_size, input_shape=10,
                 loss='mean_squared_error', n_output=1,
                 verbose=0, **kwargs):
        self._estimator_type = 'regressor'
        self.input_shape = input_shape
        self.n_output = n_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.nn_params = kwargs
        self.model = self._create_model()

    def _create_model(self):
        keras.backend.clear_session()
        return build_sequential_cnn_reg(input_shape=self.input_shape,
                                        n_output=self.n_output,
                                        loss=self.loss,
                                        **self.nn_params)

    def fit(self, X, y, reset_model=True, **kwargs):
        """
        Fit the model.

        Args
        ----------
        X: array-like
            Feature space
        y: array-like
            target space
        kwargs: dict
            arguments passed to self.model.fit()
        """
        if reset_model:
            self.model = self._create_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       verbose=self.verbose, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def set_params(self, **params):
        self.nn_params = params
        self.model = self._create_model()
        return self
