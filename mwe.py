"""Minimal working example."""

import numpy as np
from sklearn.model_selection import train_test_split

import estimator
import losses


def main():
    rng = np.random.default_rng()
    size = 100
    X = np.array(
        [rng.random(size=size), rng.integers(0, 2, size=size), rng.random(size=size),]
    ).T
    y = rng.random(size=size)
    num_idx = [0, 2]  # The indices of numerical feature columns
    cat_idx = [1]  # The indices of nominal/categorical feature columns
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Use dp root mean squared error loss for ensemble optimization
    loss = losses.DP_rMSE(privacy_budget=1.0, U=5.0)

    # Actually optimize the ensemble based on loss values
    only_useful = losses.useful_tree_predicate

    # Train the model using a depth-first approach
    model = estimator.DPGBDT(
        nb_trees=10,
        nb_trees_per_ensemble=5,
        max_depth=3,
        learning_rate=0.1,
        privacy_budget=1.0,
        loss=loss,
        use_new_tree=only_useful,
        cat_idx=cat_idx,
        num_idx=num_idx,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Prediction: {}, ground truth: {}".format(y_pred, y_test))


if __name__ == "__main__":
    main()
