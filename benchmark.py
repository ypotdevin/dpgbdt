# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# ypo@informatik.uni-kiel.de

"""Example test file."""

from functools import wraps
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn import model_selection

import estimator
import losses


def get_abalone(n_rows: Optional[int] = None) -> Any:
    """Parse the abalone dataset.

    Args:
      n_rows (int): Numbers of rows to read.

    Returns:
      Any: X, y, cat_idx, num_idx
    """
    # pylint: disable=redefined-outer-name,invalid-name
    # Re-encode gender information
    data = pd.read_csv(
        './abalone.data',
        names=['sex', 'length', 'diameter', 'height', 'whole weight',
              'shucked weight', 'viscera weight', 'shell weight', 'rings'])
    data['sex'] = pd.get_dummies(data['sex'])
    if n_rows:
        data = data.head(n_rows)
    y = data.rings.values.astype(float)
    del data['rings']
    X = data.values.astype(float)
    cat_idx = [0]  # Sex
    num_idx = list(range(1, X.shape[1]))  # Other attributes
    return X, y, cat_idx, num_idx


def cross_validate(parameters, X, y, filename):
    dummy_estimator = estimator.DPGBDT(
        nb_trees = 0,
        nb_trees_per_ensemble = 0,
        max_depth = 0,
        learning_rate= 0.0
    )
    best_model = model_selection.GridSearchCV(
        estimator = dummy_estimator,
        param_grid = parameters,
        scoring = "neg_root_mean_squared_error",
        n_jobs = 63, # leave one core free
        #n_jobs = 1, # leave one core free
        cv = model_selection.RepeatedKFold(n_splits = 5, n_repeats = 10),
        verbose = 1,
        return_train_score = True
    )
    best_model.fit(X, y)
    df = pd.DataFrame(best_model.cv_results_)
    df.to_csv(filename)


def on_abalone(cv_fun):
    """Substitute arguments `X` and `y` by Abalone data."""
    @wraps(cv_fun)
    def wrapper(**kwargs):
        parameters = kwargs.pop('parameters')
        X, y, cat_idx, num_idx = get_abalone()
        parameters = dict(
          cat_idx = [cat_idx],
          num_idx = [num_idx],
          **parameters
        )
        cv_fun(parameters = parameters, X = X, y = y, **kwargs)
    return wrapper


def with_core_parameters(cv_fun):
    """Add commonly used parameters to `parameters`."""
    @wraps(cv_fun)
    def wrapper(**kwargs):
        parameters = kwargs.pop('parameters')
        parameters = dict(
          nb_trees = [50],
          nb_trees_per_ensemble = [50],
          max_depth = [6],
          learning_rate = [0.1],
          verbosity = [0],
          **parameters
        )
        cv_fun(parameters = parameters, **kwargs)
    return wrapper


def my_cv(parameterss, filenames):
    cv_fun = cross_validate
    cv_fun = on_abalone(cv_fun)
    cv_fun = with_core_parameters(cv_fun)
    for (parameters, filename) in zip(parameterss, filenames):
        cv_fun(parameters = parameters, filename = filename)

def cv_vanilla_gbdt():
    """Setting 1"""
    cv_fun = cross_validate
    cv_fun = on_abalone(cv_fun)
    cv_fun = with_core_parameters(cv_fun)

    dfs_parameters = dict(
        privacy_budget = [None],
        clipping_bound = [None],
        only_good_trees = [False],
    )
    bfs_parameters = dict (
        use_bfs = [True],
        max_leaves = [24],
        **dfs_parameters
    )
    three_trees_parameters = dict (
        use_3_trees = [True],
        **dfs_parameters
    )

    cv_fun(parameters = dfs_parameters, filename = "vanilla_dfs.csv")
    cv_fun(parameters = bfs_parameters, filename = "vanilla_bfs.csv")
    cv_fun(parameters = three_trees_parameters, filename = "vanilla_3trees.csv")


def cv_vanilla_gbdt_with_opt():
    """Setting 2"""
    dfs_parameters = dict(
        privacy_budget = [None],
        clipping_bound = [None],
        only_good_trees = [True],
    )
    parameterss = _extend_dfs_parameters(dfs_parameters)
    filenames = [
        "vanilla_opt_dfs.csv",
        "vanilla_opt_bfs.csv",
        "vanilla_opt_3trees.csv"
    ]
    my_cv(parameterss, filenames)

def _extend_dfs_parameters(dfs_parameters):
    bfs_parameters = dict (
        use_bfs = [True],
        max_leaves = [24],
        **dfs_parameters
    )
    three_trees_parameters = dict (
        use_3_trees = [True],
        **dfs_parameters
    )
    parameterss = [
        dfs_parameters,
        bfs_parameters,
        three_trees_parameters
    ]
    return parameterss


def cv_dpgbdt():
    """Setting 3"""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = [None],
        only_good_trees = [False],
    )
    parameterss = _extend_dfs_parameters(dfs_parameters)
    filenames = [
        "dpgbdt_dfs.csv",
        "dpgbdt_bfs.csv",
        "dpgbdt_3trees.csv"
    ]
    my_cv(parameterss, filenames)


def cv_dpgbdt_with_clipping():
    """Setting 4"""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = np.logspace(-2, 1, 20),
        only_good_trees = [False],
    )
    my_cv([dfs_parameters], ["dpgbdt_clipping_dfs.csv"])


def cv_clipping():
    """Setting 5"""
    X, y, cat_idx, num_idx = get_abalone()
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    common_params = dict(
        privacy_budget = privacy_budget,
        clipping_bound = np.logspace(-2, 1, 20),
        nb_trees = [50],
        nb_trees_per_ensemble = [50],
        max_depth = [6],
        max_leaves=[24],
        learning_rate = [0.1],
        cat_idx = [cat_idx],
        num_idx = [num_idx]
    )

    dfs_parameters = common_params
    cross_validate(dfs_parameters, X, y, "dfs_5-fold-RMSE.csv")

    bfs_parameters = dict (
        use_bfs = [True],
        **common_params
    )
    cross_validate(bfs_parameters, X, y, "use_bfs_5-fold-RMSE.csv")

    three_trees_parameters = dict (
        use_3_trees = [True],
        **common_params
    )
    cross_validate(three_trees_parameters, X, y, "use_3_trees_5-fold-RMSE.csv")


def cv_dpgbdt_opt():
    """Setting 6"""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = [None],
        only_good_trees = [True],
    )
    parameterss = _extend_dfs_parameters(dfs_parameters)
    filenames = [
        "dpgbdt_opt_dfs.csv",
        #"dpgbdt_bfs.csv",
        #"dpgbdt_3trees.csv"
    ]
    my_cv([parameterss[0]], filenames)


def setting7():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = [None],
        only_good_trees = [False],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting7.csv"])


def setting8():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = [None],
        only_good_trees = [True],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting8.csv"])


def setting9():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = np.logspace(-2, 1, 20),
        only_good_trees = [False],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting9.csv"])


def setting10():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        clipping_bound = np.logspace(-2, 1, 20),
        only_good_trees = [True],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting10.csv"])

def setting11():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    losses_ = [
        losses.ExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = eps
        ) for eps in np.linspace(0.01, 0.1, 10)
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting11.csv"])

def setting12():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    losses_ = [
        losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = eps
        ) for eps in np.linspace(0.01, 0.1, 10)
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting12.csv"])

def setting13():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.RootMedianLeastSquaresError()],
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting13.csv"])

def setting14():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.RootMedianLeastSquaresError()],
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting14.csv"])

if __name__ == '__main__':
  setting12()

