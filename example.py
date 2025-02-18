# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Example test file."""

from typing import Optional, Any

import numpy as np

# pylint: disable=import-error
import pandas as pd
from sklearn.model_selection import train_test_split

# pylint: enable=import-error

import estimator


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
        "./abalone.data",
        names=[
            "sex",
            "length",
            "diameter",
            "height",
            "whole weight",
            "shucked weight",
            "viscera weight",
            "shell weight",
            "rings",
        ],
    )
    data["sex"] = pd.get_dummies(data["sex"])
    if n_rows:
        data = data.head(n_rows)
    y = data.rings.values.astype(float)
    del data["rings"]
    X = data.values.astype(float)
    cat_idx = [0]  # Sex
    num_idx = list(range(1, X.shape[1]))  # Other attributes
    return X, y, cat_idx, num_idx


if __name__ == "__main__":
    # pylint: disable=redefined-outer-name,invalid-name
    X, y, cat_idx, num_idx = get_abalone()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # A simple baseline: mean of the training set
    y_pred = np.mean(y_train).repeat(len(y_test))
    print("Mean - RMSE: {0:f}".format(np.sqrt(np.mean(np.square(y_pred - y_test)))))

    # Train the model using a depth-first approach
    model = estimator.DPGBDT(
        privacy_budget=0.1,
        nb_trees=50,
        nb_trees_per_ensemble=50,
        max_depth=6,
        learning_rate=0.1,
        cat_idx=cat_idx,
        num_idx=num_idx,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        "Depth first growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )

    # Train the model using a best-leaf first approach
    model = estimator.DPGBDT(
        privacy_budget=0.1,
        nb_trees=50,
        nb_trees_per_ensemble=50,
        max_depth=6,
        max_leaves=24,
        learning_rate=0.1,
        use_bfs=True,
        cat_idx=cat_idx,
        num_idx=num_idx,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        "Best-leaf first growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )

    # Train the model using 3-nodes trees combination approach
    model = estimator.DPGBDT(
        privacy_budget=0.1,
        nb_trees=50,
        nb_trees_per_ensemble=50,
        max_depth=6,
        learning_rate=0.1,
        use_3_trees=True,
        cat_idx=cat_idx,
        num_idx=num_idx,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(
        "3-nodes trees growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )
