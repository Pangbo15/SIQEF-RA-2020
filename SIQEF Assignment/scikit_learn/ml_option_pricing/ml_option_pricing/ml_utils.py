import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.autonotebook import tqdm
from hyperopt import Trials
from hyperopt import STATUS_OK
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import f1_score,\
    roc_auc_score,\
    log_loss,\
    mean_squared_error,\
    mean_absolute_error
from hyperopt import fmin, tpe


built_in_evalmetric_info = {
                    'f1': True,
                    'auc': True,
                    'mae': False,
                    'mse': False,
                    'rmse': False
                  }


def f1(y_true, y_pred, **kwargs):
    """
    compute f1 score
    :param y_true: array-like of true values
    :param y_pred: ndarray-like of predicted values
    :param kwargs: any other argument of sklearn.metrics.f1_score
    :return: scalar
    Note : return should be in 0 1
    """
    if type_of_target(y_pred).endswith('multioutput') or\
            type_of_target(y_pred).startswith('multilabel'):
        y_pred = np.argmax(y_pred, axis=1)

    return f1_score(y_true, y_pred, average='weighted', **kwargs)


def auc(y_true, y_pred, **kwargs):
    """
    AUC
    :param y_true: array-like of true values
    :param y_pred: array-like of predicted values
    :param kwargs: any other argument of sklearn.metrics.auc
    :return: scalar
    Note : does not accept proba-like prediction
    """
    if type_of_target(y_pred).endswith('multioutput') or \
            type_of_target(y_pred).startswith('multilabel'):
        y_pred = np.argmax(y_pred, axis=1)

    return roc_auc_score(y_true, y_pred, **kwargs)


def mae(y_true, y_pred, **kwargs):
    """
    Mean absolute error
    :param y_true: array-like
    :param y_pred: array-like
    :param kwargs: any other argument of sklearn.metrics.mean_absolute_error
    :return: positive scalar
    """
    y_pred = np.nan_to_num(y_pred)
    return mean_absolute_error(y_true, y_pred, **kwargs)


def mse(y_true, y_pred, **kwargs):
    """
    Mean absolute error
    :param y_true: array-like
    :param y_pred: array-like
    :param kwargs: any other argument of sklearn.metrics.mean_absolute_error
    :return: positive scalar
    """
    y_pred = np.nan_to_num(y_pred)
    return mean_squared_error(y_true, y_pred, **kwargs)


def rmse(y_true, y_pred, **kwargs):
    """
    Root mean squared error
    :param y_true: array-like
    :param y_pred: array-like
    :param kwargs: any other argument of sklearn.metrics.mean_squared_error
    :return: positive scalar
    """
    return np.sqrt(mean_squared_error(y_true, y_pred, **kwargs))


def apply_evalmetric(metric_name):

    if metric_name == 'f1':
        return f1
    elif metric_name == 'auc':
        return auc
    elif metric_name == 'mae':
        return mae
    elif metric_name == 'mse':
        return mse
    elif metric_name == 'rmse':
        return rmse
    else:
        raise ValueError('unknown {} metric'.format(metric_name))


def _build_sample_weight(weight_feature, scaling='minmax'):
    """
    Scaling sample weight values
    :param weight_feature: array-like
    :return: 1d array
    """
    if scaling == 'minmax':
        scaler = MinMaxScaler()
        weight_feature = np.absolute(weight_feature)
        weights = scaler.fit_transform(weight_feature.reshape(-1, 1))
        return np.ravel(weights)
    elif scaling == 'mean':
        return (np.absolute(weight_feature) / np.absolute(weight_feature).sum())\
               * len(weight_feature)


def set_params(model, **params):
    if hasattr(model, 'set_params'):
        model.set_params(**params)
    else:
        for param, val in params.items():
            setattr(model, param, val)


def bayesian_tuning(X, y, model, param_grid, loss_metric,
                    n_kfold=5, y_transform=None, static_params={},
                    sample_weight=None, trials=Trials,
                    nb_evals=50, optimizer=tpe.suggest,
                    **kwargs):
    """
    Performe an Bayesian optimization on given ML model and store trials in an dict.

    :param X: pd.DataFrame, X data
    :param y: pd.DataFrame or pd.Series, y data
    :param model: instance of ML model implementing fit, predict and set_params methods
    :param param_grid: Hyperopt type grid search dictionary (see Hyperopt doc :
            https://github.com/hyperopt/hyperopt/wiki/FMin)
    :param loss_metric: str, refer to the name of available computed
            model metrics {'f1', 'auc', 'mae', 'rmse'}
             or callable which yield (score_name(str), score(float), is_higher_better(bool))
    :param n_kfold: (int) number of folds for cross validation during search
    :param static_params: model hyperparameter that are not tuned
    :param sample_weight: pd.Series, values for model sample weight
    :param trials : type of Database used for hp calibration, can be Hyperopt
            Trials object or MongoTrials object
    :param nb_evals: number of iteration of optimization process
    :param optimizer: optimizer used by hyperopt
    :param kwargs: any other parameters passed to model.fit()

    :return: list of dict containing optimization info at each iteration
    """

    pbar = tqdm(total=nb_evals,
                desc="{} hyper optim".format(model.__class__.__name__),
                file=sys.stdout)

    def weighted_mean_folds(data, weights):
        """function for weights averaging on cv test fold """
        data = data.dropna(axis=1)
        wm = np.average(data.values, axis=0, weights=weights)
        res = {}
        for i in range(len(data.columns)):
            res[data.columns[i]] = wm[i]
        return res

    def objective(hyperparameters):
        """Objective function for hyperopt optimization. Returns
           the cross validation score from a set of hyperparameters."""

        pbar.update(1)
        global ITERATION
        ITERATION += 1

        all_params = {**hyperparameters, **static_params}
        set_params(model, **all_params)

        result_score = kfold_cv(
            model=model,
            X=X,
            y=y,
            n_kfold=n_kfold,
            y_transform=y_transform,
            sample_weight=sample_weight,
            loss_metric=loss_metric,
            **kwargs,
        )

        # compute weighted mean on test folds, default weights set to one
        weights = np.ones(len(result_score))
        agg_score = weighted_mean_folds(result_score, weights)
        agg_score['hyperparameters'] = all_params
        agg_score['status'] = STATUS_OK
        agg_score['iteration'] = ITERATION

        return agg_score

    global ITERATION
    ITERATION = 0

    trials = trials()
    results = fmin(fn=objective, space=param_grid, algo=optimizer, trials=trials,
                  max_evals=nb_evals)
    pbar.close()
    trials_dict = sorted(trials.results, key=lambda x: x['loss'])

    return trials_dict


def kfold_cv(X, y, model,
             n_kfold=5,
             y_transform=None,
             sample_weight=None,
             loss_metric=None,
             **kwargs):
    """
    Perform k-fold cross-validation
    :param X: array_like
    :param y: array-like
    :param model: instance of ML class implementing fit(), predict()
    :param sample_weight: pd.Series, values used as sample weights in model.fit()
    :param loss_metric: str in {'mae', 'mse', 'rmse', 'f1'}
    :param kwargs: passed to `model.fit`

    :return: pd.DataFrame with score for each folds
    """
    kf = KFold(n_splits=n_kfold, shuffle=False)
    cv_score = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        if sample_weight is not None:
            W_train = _build_sample_weight(sample_weight.values[train_index])
            W_test = _build_sample_weight(sample_weight.values[test_index])
        else:
            W_train, W_test = None, None

        model.fit(X_train, y_train, sample_weight=W_train, **kwargs)
        y_pred = model.predict(X_test)

        kf_score = {}
        if callable(loss_metric):
            _, loss, is_higher_better = loss_metric(y_test, y_pred)
            kf_score['loss'] = (-1 if is_higher_better else 1) * loss

        elif loss_metric in built_in_evalmetric_info.keys():
            loss = apply_evalmetric(loss_metric)(y_test,
                                                 y_pred,
                                                 sample_weight=W_test)
            is_higher_better = built_in_evalmetric_info[loss_metric]
            kf_score['loss'] = (-1 if is_higher_better else 1) * loss

        else:
            raise ValueError('unknown loss_metric')

        cv_score.append(kf_score)

    return pd.DataFrame(cv_score)


def serialize(estimators, features,
              fit_params, filename,
              creation_date=datetime.now()):
    """
    save fitted model
    :param estimators: list of fitted model
    :param features: list of feature name used
    :param fit_params: dic of parameter used to fit model
    :param filename: path to file
    :param creation_date: Timestamp default now
    :return:
    """
    data = {'estimator': estimators,
            'features': features,
            'fit_params': fit_params,
            'creation_date': creation_date}
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_model(filename):
    """load saved model"""
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
