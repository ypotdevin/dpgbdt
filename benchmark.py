# gtheo@ethz.ch
# ypo@informatik.uni-kiel.de

"""Example test file."""

import argparse
import logging
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import compose, metrics, model_selection, pipeline, preprocessing
from sklearn.preprocessing import MinMaxScaler

import estimator
import losses

DUMMY_ESTIMATOR = estimator.DPGBDT(
    nb_trees = 0,
    nb_trees_per_ensemble = 0,
    max_depth = 0,
    learning_rate = 0.0
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
        './training_data/abalone/abalone.data',
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

def get_adult() -> Tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    df = pd.read_csv('training_data/adult/adult_train.csv')
    df = df.replace('?', np.nan)
    df = df.dropna(axis = 0, how = "any")

    nomi_steps = [('onehot', preprocessing.OneHotEncoder(sparse = False))]
    nomi_pipe = pipeline.Pipeline(nomi_steps)
    nomi_cols = [
        'workclass', 'marital.status', 'occupation', 'relationship', 'race',
        'sex', 'native.country'
    ]
    transformers = [
        ('nominal', nomi_pipe, nomi_cols),
        ('numeric', 'passthrough', ['age', 'capital.gain', 'capital.loss',
                                    'hours.per.week'] )
    ]
    # This implicitly drops the columns 'fnlwgt', 'education' and
    # 'income'
    col_trans = compose.ColumnTransformer(transformers = transformers)
    X = col_trans.fit_transform(df)

    y_enc = preprocessing.LabelEncoder()
    y = y_enc.fit_transform(df['income'])

    cat_idx = list(range(82)) # Assume the first 82 columns are created
                              # by the OneHotEncoder based on the 7
                              # input features
    num_idx = list(range(82, 86))
    return X, y, cat_idx, num_idx

def get_MSD() -> Tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    df = pd.read_csv('training_data/MSD/year_prediction.csv')
    X = df.drop(labels = 'label', axis = 1).values # timbre averages and covariances (floats)
    X = X[:463715] # The remaining ones are the final test set
    cat_idx = [] # type: ignore
    num_idx = list(range(X.shape[1]))
    y = df['label'].values # release year (int64)
    y = y[:463715]
    return X, y, cat_idx, num_idx

def cross_validate(
        parameters: dict[str, Any],
        X, y,
        filename: str,
        dummy_estimator,
        parameter_preprocessing: Callable[[dict[str, Any]], dict[str, Any]],
        n_jobs: int,
        cv = None,
    ) -> None:
    if cv is None:
        cv = model_selection.RepeatedKFold(n_splits = 5, n_repeats = 10)
    parameters = parameter_preprocessing(parameters)
    best_model = model_selection.GridSearchCV(
        estimator = dummy_estimator,
        param_grid = parameters,
        scoring = "neg_root_mean_squared_error",
        n_jobs = n_jobs,
        cv = cv,
        return_train_score = True
    )
    best_model.fit(X, y)
    df = pd.DataFrame(best_model.cv_results_)
    df.to_csv(filename)


def on_dataset(
        cv_fun: Callable[..., None],
        which: str
    ) -> Callable[..., None]:
    @wraps(cv_fun)
    def wrapper(**kwargs: dict[str, Any]) -> None:
        parameters = kwargs.pop('parameters')
        if which == 'abalone':
            get_ds = get_abalone
        elif which == 'adult':
            get_ds = get_adult
        elif which == 'MSD':
            get_ds = get_MSD
        else:
            raise ValueError("Unsupported dataset: {}".format(which))
        X, y, cat_idx, num_idx = get_ds()
        parameters = dict(
          cat_idx = [cat_idx],
          num_idx = [num_idx],
          **parameters
        )
        cv_fun(parameters = parameters, X = X, y = y, **kwargs)
    return wrapper

def with_basic_estimator_config(
        cv_fun: Callable[..., None]
    ) -> Callable[..., None]:
    """Set commonly used parameters."""
    @wraps(cv_fun)
    def wrapper(**kwargs: dict[str, Any]) -> None:
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

def with_io_params(
        cv_fun: Callable[..., None],
        dataset: str,
        n_jobs: int,
    ) -> Callable[..., None]:
    @wraps(cv_fun)
    def wrapper(**kwargs: dict[str, Any]) -> None:
        cv_fun(n_jobs = n_jobs, **kwargs)
    return on_dataset(wrapper, which = dataset)

def my_cv2(
        *,
        parameters: dict[str, Any],
        filename: str,
        dataset: str = 'abalone',
        n_jobs: int = 60,
        pipeline = None, # type: ignore
        parameter_preprocessing: Callable[[dict[str, Any]], dict[str, Any]] = None,
    ) -> None:
    if pipeline is None:
        pipeline = DUMMY_ESTIMATOR
    if parameter_preprocessing is None:
        parameter_preprocessing = lambda x: x
    cv_fun = cross_validate
    cv_fun = with_basic_estimator_config(cv_fun)
    cv_fun = with_io_params(cv_fun, dataset, n_jobs)
    cv_fun = with_pipeline_params(cv_fun, pipeline, parameter_preprocessing)
    cv_fun(parameters = parameters, filename = filename)

def with_pipeline_params(
        cv_fun: Callable[..., None],
        pipeline,
        parameter_preprocessing: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[..., None]:
    @wraps(cv_fun)
    def wrapper(**kwargs: dict[str, Any]) -> None:
        cv_fun(
            dummy_estimator = pipeline,
            parameter_preprocessing = parameter_preprocessing,
            **kwargs
        )
    return wrapper

def my_cv(
        parameterss: Iterable[dict[str, Any]],
        filenames: Iterable[str],
        n_jobs: int = 60,
        dummy_estimator = None,
        parameter_preprocessing = None):
    if dummy_estimator is None:
        dummy_estimator = DUMMY_ESTIMATOR
    if parameter_preprocessing is None:
        parameter_preprocessing = lambda x: x

    cv_fun = cross_validate
    cv_fun = on_dataset(cv_fun, which = 'abalone')
    cv_fun = with_basic_estimator_config(cv_fun)
    for (parameters, filename) in zip(parameterss, filenames):
        cv_fun(
            parameters = parameters,
            filename = filename,
            n_jobs = n_jobs,
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
    return pipeline.make_pipeline(MinMaxScaler(), final_estimator)


def cv_vanilla_gbdt() -> None:
    """Setting 1"""
    cv_fun = cross_validate
    cv_fun = on_dataset(cv_fun, which = 'abalone')
    cv_fun = with_basic_estimator_config(cv_fun)

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


def cv_vanilla_gbdt_with_opt() -> None:
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


def cv_dpgbdt() -> None:
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


def cv_dpgbdt_with_clipping() -> None:
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


def cv_clipping() -> None:
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


def cv_dpgbdt_opt() -> None:
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


def setting7() -> None:
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


def setting8() -> None:
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


def setting9() -> None:
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


def setting10() -> None:
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

def setting11() -> None:
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

def setting11a() -> None:
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

def setting12() -> None:
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

def setting13() -> None:
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

def setting14() -> None:
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

def setting15() -> None:
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

def setting16() -> None:
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

def setting17() -> None:
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

def setting18() -> None:
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

def setting19() -> None:
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

def setting20() -> None:
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

def setting21() -> None:
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.DP_quasi_rMSE(
            privacy_budget = 1,
            beta = 0.5,
            L = -100,
            U = 100,
            m = 0,
            seed = 42,
        )
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
        filenames = ["setting21.csv"],
    )

def setting22(n_jobs: int) -> None:
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.DP_rMSE(
            privacy_budget = 1.0,
            U = 40,
            seed = 42,
        )
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
        filenames = ["setting22.csv"],
        n_jobs = n_jobs,
    )

def setting23(n_jobs: int) -> None:
    """See jupyter notebook for more explanation."""
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.LeastSquaresError()
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv(
        parameterss = [dfs_parameters],
        filenames = ["setting23.csv"],
        n_jobs = n_jobs,
    )

def setting24(n_jobs: int) -> None:
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.LeastSquaresError()
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv2(
        parameters = dfs_parameters,
        filename = "setting24.csv",
        dataset = 'MSD',
        n_jobs = n_jobs,
    )

def setting25(n_jobs: int) -> None:
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.DP_rMSE(
            privacy_budget = 1,
            U = 100, # TODO
            seed = 42,
        )
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv2(
        parameters = dfs_parameters,
        filename = "setting25.csv",
        dataset = 'MSD',
        n_jobs = n_jobs,
    )

def setting26(n_jobs: int) -> None:
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.LeastSquaresError()
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.keep_each_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv2(
        parameters = dfs_parameters,
        filename = "setting26.csv",
        dataset = 'adult',
        n_jobs = n_jobs,
    )

def setting27(n_jobs: int) -> None:
    privacy_budget = np.append(
        np.linspace(0.1, 0.9, num = 9),
        np.linspace(1.0, 5.0, num = 9),
    )
    losses_ = [
        losses.DP_rMSE(
            privacy_budget = 1,
            U = 100, # TODO
            seed = 42,
        )
    ]
    dfs_parameters = dict(
        privacy_budget = privacy_budget,
        loss = losses_,
        use_new_tree = [losses.useful_tree_predicate],
        gradient_filtering = [True],
        leaf_clipping = [True],
    )
    my_cv2(
        parameters = dfs_parameters,
        filename = "setting27.csv",
        dataset = 'adult',
        n_jobs = n_jobs,
    )

def single_run(
        get_dataset,
        loss,
        predicate,
        further_model_params,
    ) -> None:
    X, y, cat_idx, num_idx = get_dataset()
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y)
    model_params = dict(
        nb_trees = 50,
        nb_trees_per_ensemble = 50,
        max_depth = 6,
        learning_rate = 0.1,
        privacy_budget = 1.0,
        gradient_filtering = True,
        leaf_clipping = True,
    )
    model_params.update(further_model_params)
    model = estimator.DPGBDT(
        loss = loss,
        use_new_tree = predicate,
        cat_idx=cat_idx,
        num_idx=num_idx,
        **model_params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = metrics.mean_squared_error(
        y_true = y_val, y_pred = y_pred, squared = False
    )
    print(
        "rMSE of DPGBDT with loss {} and predicate {}: {}".format(
            loss, predicate, rmse
        )
    )

SETTING_DISPATCH = {
    22 : setting22,
    23 : setting23,
    24 : setting24,
    25 : setting25,
    26 : setting26,
    27 : setting27,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Start a single run.")
    parser.add_argument(
        "setting", metavar='SETTING', type=int,
        help = 'The setting to execute.'
    )
    parser.add_argument(
        '-U', type = float, default = 100.0,
        help = "Upper bound on prediction differences"
    )
    parser.add_argument(
        '--nb-trees', type = int, default = 50,
        help = "Total number of trees in total."
    )
    parser.add_argument(
        '--nb-trees-per-ensemble', type = int, default = 50,
        help = "Number of trees per ensemble."
    )
    parser.add_argument(
        '--max-depth', type = int, default = 60,
        help = "Maximal depth for every single tree."
    )
    parser.add_argument(
        '--n-jobs', type = int, default = 60,
        help = "Number of (logical) cores to use for cross validation."
    )
    args = parser.parse_args()

    f = SETTING_DISPATCH[args.setting]
    logging.basicConfig(
       filename='{}.log'.format(f.__name__),
       level = logging.INFO
    )
    f(n_jobs = args.n_jobs)

    # logging.basicConfig(
    #    level = logging.DEBUG
    # )
    # single_run(
    #     get_abalone,
    #     losses.DP_rMSE(privacy_budget = 1.0, U = args.U),
    #     losses.useful_tree_predicate,
    #     dict(
    #         nb_trees = args.nb_trees,
    #         nb_trees_per_ensemble = args.nb_trees_per_ensemble,
    #         max_depth = args.max_depth
    #     )
    # )
