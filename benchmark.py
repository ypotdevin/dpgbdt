# gtheo@ethz.ch
# ypo@informatik.uni-kiel.de

"""Example test file."""

import logging
import sys
from functools import wraps
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

import estimator
import losses

DUMMY_ESTIMATOR = estimator.DPGBDT(
    nb_trees = 0,
    nb_trees_per_ensemble = 0,
    max_depth = 0,
    learning_rate= 0.0
)


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


def cross_validate(
        parameters: dict[str, Any],
        X, y,
        filename: str,
        dummy_estimator,
        parameter_preprocessing: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    parameters = parameter_preprocessing(parameters)
    best_model = model_selection.GridSearchCV(
        estimator = dummy_estimator,
        param_grid = parameters,
        scoring = "neg_root_mean_squared_error",
        n_jobs = 63, # leave one core free
        #n_jobs = 1,
        cv = model_selection.RepeatedKFold(n_splits = 5, n_repeats = 10),
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
          **parameters
        )
        cv_fun(parameters = parameters, **kwargs)
    return wrapper


def my_cv(
        parameterss: Iterable[dict[str, Any]],
        filenames: Iterable[str],
        dummy_estimator = None,
        parameter_preprocessing = None):
    if dummy_estimator is None:
        dummy_estimator = DUMMY_ESTIMATOR
    if parameter_preprocessing is None:
        parameter_preprocessing = lambda x: x

    cv_fun = cross_validate
    cv_fun = on_abalone(cv_fun)
    cv_fun = with_core_parameters(cv_fun)
    for (parameters, filename) in zip(parameterss, filenames):
        cv_fun(
            parameters = parameters,
            filename = filename,
            dummy_estimator = dummy_estimator,
            parameter_preprocessing = parameter_preprocessing
        )

def to_pipeline_params(params: dict[str, Any], step_prefix: str):
    """Rename estimator parameters to be usable an a pipeline.

    See https://scikit-learn.org/stable/modules/compose.html#pipeline.
    """
    return {
        "{0}__{1}".format(step_prefix, key) : value
            for key, value in params.items()
    }

def min_max_pipeline(final_estimator):
    return make_pipeline(MinMaxScaler(), final_estimator)


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
    cv_fun(
        parameters = three_trees_parameters,
        filename = "vanilla_3trees.csv"
    )


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
        losses.RootExpQLeastSquaresError(
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

def setting11a():
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
        ) for eps in np.logspace(-1, 0, 20)
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv([dfs_parameters], ["setting11a.csv"])

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

def setting15():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9)
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.LeastSquaresError()],
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True]
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting15.csv"],
        dummy_estimator = min_max_pipeline(DUMMY_ESTIMATOR),
        parameter_preprocessing = lambda x: to_pipeline_params(x, 'dpgbdt')
    )

def setting16():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.LeastSquaresError()],
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting16.csv"],
    )

def setting17():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.RootMedianLeastSquaresError()],
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting17.csv"],
    )

def setting18():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = 0.1
        )],
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting18.csv"],
    )

def setting18():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = [losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = 0.1
        )],
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting18.csv"],
    )

def setting19():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = 0.1,
            q = q,
        ) for q in [0.6, 0.7, 0.8, 0.9]
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting19.csv"],
    )

def setting20():
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.RootQuantileLeastSquaresError(q = q)
        for q in [0.6, 0.7, 0.8, 0.9]
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting20.csv"],
    )

def in_depth_analysis_1():
    X, y, cat_idx, num_idx = get_abalone()
    model = estimator.DPGBDT(
        nb_trees = 50,
        nb_trees_per_ensemble = 50,
        max_depth = 6,
        learning_rate = 0.1,
        privacy_budget = 1.0,
        loss = losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = 0.1
        ),
        use_new_tree = losses.useful_tree_predicate,
        gradient_filtering = True,
        leaf_clipping = True,
        cat_idx=cat_idx,
        num_idx=num_idx
    )
    model.fit(X, y)

def in_depth_analysis_2():
    X, y, cat_idx, num_idx = get_abalone()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    model = estimator.DPGBDT(
        nb_trees = 50,
        nb_trees_per_ensemble = 50,
        max_depth = 6,
        learning_rate = 0.1,
        privacy_budget = 1.0,
        loss = losses.RootExpQLeastSquaresError(
            lower_bound = 0.0,
            upper_bound = 100.0,
            privacy_budget = 0.1,
            q = 0.9
        ),
        use_new_tree = losses.useful_tree_predicate,
        gradient_filtering = True,
        leaf_clipping = True,
        cat_idx = cat_idx,
        num_idx = num_idx
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean(np.square(y_pred - y_test)))
    print('Depth first growth - RMSE: {0:f}'.format(rmse))
    return rmse

if __name__ == '__main__':
    f = setting20
    logging.basicConfig(
       filename='{}.log'.format(f.__name__),
       encoding='utf-8',
       level = logging.DEBUG
    )
    f()
