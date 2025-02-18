# -*- coding: utf-8 -*-
# gtheo@ethz.ch
"""Implement ensemble of differentially private gradient boosted trees.

From: https://arxiv.org/pdf/1911.04209.pdf
"""

import math
import operator
from collections import defaultdict
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

import logging
from losses import (
    ClassificationLossFunction,
    LeastSquaresError,
    LossFunction,
    keep_each_tree_predicate,
)

logger = logging.getLogger(__name__)


class GradientBoostingEnsemble:
    """Implement gradient boosting ensemble of trees.

  Attributes:
    trees (List[List[DifferentiallyPrivateTree]]): A list of DP boosted k-class trees.
  """

    # pylint: disable=invalid-name, too-many-arguments, unused-variable

    def __init__(
        self,
        nb_trees: int,
        nb_trees_per_ensemble: int,
        n_classes: Optional[int] = None,
        max_depth: int = 6,
        privacy_budget: Optional[float] = None,
        loss: Union[LossFunction, ClassificationLossFunction] = None,
        l2_threshold: float = 1.0,
        l2_lambda: float = 0.1,
        use_new_tree: Callable[[Any, Any, float, float], bool] = None,
        learning_rate: float = 0.1,
        early_stop: int = 5,
        max_leaves: Optional[int] = None,
        min_samples_split: int = 2,
        gradient_filtering: bool = False,
        leaf_clipping: bool = False,
        balance_partition: bool = True,
        use_bfs: bool = False,
        use_3_trees: bool = False,
        use_decay: bool = False,
        splitting_grid: Optional[Sequence[Any]] = None,
        cat_idx: Optional[List[int]] = None,
        num_idx: Optional[List[int]] = None,
    ) -> None:
        """Initialize the GradientBoostingEnsemble class.

    Args:
      nb_trees (int): The total number of trees in the model.
      nb_trees_per_ensemble (int): The number of trees in each ensemble.
      n_classes (int): Number of classes. Triggers regression (None) vs
          classification.
      max_depth (int): Optional. The depth for the trees. Default is 6.
      privacy_budget (float): Optional. The privacy budget available for the
          model. Default is 1.0. If `None`, do not apply differential privacy.
      loss (Union[LossFunction, ClassificationLossFunction]): A loss object
          (preferably from the `losses` module), which defines primarily how to
          compute the loss of the ensemble (e.g. by using regular MSE, clipped
          MSE, median, DP-median, ...). Default is (leaky) LSE.
      l2_threshold (float):
          Threshold for the loss function. For the square loss function
          (default), this is 1.0.
      l2_lambda (float):
          Regularization parameter for l2 loss function. For the square
          loss function (default), this is 0.1.
      use_new_tree (Callable[[Any, Any, float, float], bool]): A predicate which
          compares the previous loss (first float) to the current loss (second
          float; including the newly created decision tree) and decides whether
          to keep the new tree (True) or to discard it (False). It may base its
          decision additionally on the arguments (first and second Any) provided
          to the loss function, which calculated the previous and the current
          loss mentioned above. By default keep each new tree.
      learning_rate (float): Optional. The learning rate. Default is 0.1.
      early_stop (int): Optional. If the ensemble loss doesn't decrease for
          <int> consecutive rounds, abort training. Default is 5. Has no effect,
          if `only_good_trees` is False or if DP is disabled.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      gradient_filtering (bool): Optional.
          Whether or not to perform gradient based data filtering during
          training (only available on regression). Default is False.
      leaf_clipping (bool): Optional.
          Whether or not to clip the leaves after training (only
          available on regression). Default is False.
      balance_partition (bool): Optional. Balance data repartition for training
          the trees. The default is True, meaning all trees within an ensemble
          will receive an equal amount of training samples. If set to False,
          each tree will receive <x> samples where <x> is given in line 8 of
          the algorithm in the author's paper.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      use_decay (bool): Optional. If True, internal node privacy budget has a
          decaying factor.
      splitting_grid (Sequence[np.array]): Optional.
          If provided, use these (data independent) per feature
          splitting candidates for all trees, to find the best value to
          split on while building any decision tree. It is assumed, but
          not checked, that each feature array conains no duplicates.
          If not provided, use the data feature values to split on.
      cat_idx (List): Optional.
          List of indices of nominal or categorical feature columns. If
          `None`, assume no features are nomi
      num_idx (List): Optional.
          List of indices for numerical feature columns.
      """
        self.nb_trees = nb_trees
        if privacy_budget is not None and nb_trees_per_ensemble > nb_trees:
            raise ValueError(
                "Number of trees per ensemble may be at most the total "
                "number of trees."
            )
        self.nb_trees_per_ensemble = nb_trees_per_ensemble
        if n_classes is not None:
            raise ValueError("Classification is currently not supported (leaky).")
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.privacy_budget = privacy_budget
        self.loss = loss
        self.l2_threshold = l2_threshold
        self.l2_lambda = l2_lambda
        self.use_new_tree = use_new_tree
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.gradient_filtering = gradient_filtering
        self.leaf_clipping = leaf_clipping
        self.balance_partition = balance_partition
        if use_bfs:
            raise ValueError("The option `use_bfs` is not supported anymore.")
        self.use_bfs = use_bfs
        if use_3_trees:
            raise ValueError("The option `use_3_trees` is not supported anymore.")
        self.use_3_trees = use_3_trees
        self.use_decay = use_decay
        self.splitting_grid = splitting_grid
        self.cat_idx = cat_idx
        self.num_idx = num_idx

        if self.privacy_budget is None or self.privacy_budget == 0.0:
            logger.info("No privacy budget provided. Differential privacy is disabled.")
            # TODO
            logger.warn(
                "There are indications, that the implementation of GBDT without "
                "differential privacy contains errors."
            )
            self.use_dp = False
        else:
            if self.privacy_budget > 100:
                logger.warning(
                    "High privacy budget detected (budget > 100). This affects the "
                    "exponential mechanism badly. Note that the budget is limited to 1000 "
                    "anyway."
                )
            self.use_dp = True
            self.privacy_budget = min(self.privacy_budget, 1000)

        if self.loss is None:
            self.loss = LeastSquaresError()

        if self.use_new_tree is None:
            self.use_new_tree = keep_each_tree_predicate

        if self.use_3_trees and self.use_bfs:
            # Since we're building 3-node trees it's the same anyways.
            self.use_bfs = False

        # The inner list represents k-trees (singletons for k = 1), the outer list
        # is the spine of the boosted trees.
        self.trees = []  # type: List[List[DifferentiallyPrivateTree]]

        self.init_ = self.loss.init_estimator()

        # Initial score
        self.init_score = None

        # This handles attribute comparison depending on the attribute's nature
        self.feature_to_op = defaultdict(
            lambda: (operator.lt, operator.ge)
        )  # type: Dict[int, Any]
        if self.cat_idx:
            for feature_index in self.cat_idx:
                self.feature_to_op[feature_index] = (operator.eq, operator.ne)

    def Train(self, X: np.array, y: np.array) -> "GradientBoostingEnsemble":
        """Train the ensembles of gradient boosted trees.

    Args:
      X (np.array): The features.
      y (np.array): The label.

    Returns:
      GradientBoostingEnsemble: A GradientBoostingEnsemble object.
    """

        # Init gradients
        self.init_.fit(X, y)
        self.init_score = self.loss.get_init_raw_predictions(
            X, self.init_
        )  # (n_samples, K)
        logger.debug("Training initialized with score: {}".format(self.init_score))
        update_gradients = True

        # Number of ensembles in the model
        nb_ensembles = int(np.ceil(self.nb_trees / self.nb_trees_per_ensemble))
        logger.info("Model will have {0:d} ensembles".format(nb_ensembles))

        if self.use_dp:
            # Privacy budget allocated to all trees in each ensemble
            tree_privacy_budget = np.divide(self.privacy_budget, nb_ensembles)
            # In multi-class classification the budget for each tree
            # is the same as for the whole K trees but halved
            # As each data point is only assigned to one class,
            # it only matters if it is assigned to the considered class or not but not
            # to which other
            # Thus it always remains 2 - independently of how many total classes exists
            privacy_budget_per_tree = 2 if self.loss.is_multi_class else 1
            tree_privacy_budget = np.divide(
                tree_privacy_budget, privacy_budget_per_tree
            )

        prev_loss = np.inf
        early_stop_round = self.early_stop

        # Train all trees
        for tree_index in range(self.nb_trees):
            # Compute sensitivity
            delta_g = 3 * np.square(self.l2_threshold)
            delta_v = self.l2_threshold / (1 + self.l2_lambda)
            if self.leaf_clipping:
                delta_v = min(
                    delta_v,
                    2
                    * self.l2_threshold
                    * math.pow((1 - self.learning_rate), tree_index),
                )

            current_tree_for_ensemble = tree_index % self.nb_trees_per_ensemble
            if current_tree_for_ensemble == 0:
                # Initialize the dataset and the gradients
                X_ensemble = np.copy(X)
                y_ensemble = np.copy(y)
                prev_loss = np.inf
                update_gradients = True
                # gradient initialization will happen later in the per-class-loop

            if self.use_dp:
                # Compute the number of rows that the current tree will use for training
                if self.balance_partition:
                    # All trees will receive same amount of samples
                    if self.nb_trees % self.nb_trees_per_ensemble == 0:
                        # Perfect split
                        number_of_rows = int(len(X) / self.nb_trees_per_ensemble)
                    else:
                        # Partitioning data across ensembles
                        if np.ceil(tree_index / self.nb_trees_per_ensemble) == np.ceil(
                            self.nb_trees / self.nb_trees_per_ensemble
                        ):
                            number_of_rows = int(
                                len(X) / (self.nb_trees % self.nb_trees_per_ensemble)
                            )
                        else:
                            number_of_rows = int(
                                len(X) / self.nb_trees_per_ensemble
                            ) + int(
                                len(X) / (self.nb_trees % self.nb_trees_per_ensemble)
                            )
                else:
                    # Line 8 of Algorithm 2 from the paper
                    number_of_rows = int(
                        (
                            len(X)
                            * self.learning_rate
                            * math.pow(
                                (1 - self.learning_rate), current_tree_for_ensemble
                            )
                        )
                        / (
                            1
                            - math.pow(
                                (1 - self.learning_rate), self.nb_trees_per_ensemble
                            )
                        )
                    )

                # If using the formula from the algorithm, some trees may not get
                # samples. In that case we skip the tree and issue a warning. This
                # should hint the user to change its parameters (likely the ensembles
                # are too unbalanced)
                if number_of_rows == 0:
                    logger.warning(
                        "The choice of trees per ensemble vs. the total number"
                        " of trees is not balanced properly; some trees will "
                        "not get any training samples. Try using "
                        "balance_partition=True or change your parameters."
                    )
                    continue

                ### # Select <number_of_rows> rows at random from the ensemble dataset
                ### rows = np.random.randint(len(X_ensemble), size=number_of_rows)

                # train for each class a separate tree on the same rows.
                # In regression or binary classification, K has been set to one.
                k_trees = []  # type: List[DifferentiallyPrivateTree]
                for kth_tree in range(self.loss.K):
                    if tree_index == 0:
                        # First tree, start with initial scores (mean of labels)
                        assert self.init_score is not None
                        gradients = self.ComputeGradientForLossFunction(
                            y, self.init_score[: len(y)], kth_tree
                        )
                    else:
                        # Update gradients of all training instances on loss l
                        if update_gradients:
                            gradients = self.ComputeGradientForLossFunction(
                                y_ensemble, self.Predict(X_ensemble), kth_tree
                            )  # type: ignore

                    assert gradients is not None
                    split_data = self._tree_data_sample_split(
                        gradients=gradients,
                        X_ensemble=X_ensemble,
                        y_ensemble=y_ensemble,
                        n_samples_per_tree=number_of_rows,
                    )
                    gradients_tree = split_data["gradients_tree"]
                    X_tree = split_data["X_tree"]
                    y_tree = split_data["y_tree"]

                    logger.debug(
                        "Tree {0:d} will receive a budget of epsilon={1:f} and "
                        "train on {2:d} instances.".format(
                            tree_index, tree_privacy_budget, len(X_tree)
                        )
                    )
                    # Fit a differentially private decision tree
                    tree = DifferentiallyPrivateTree(
                        tree_index,
                        self.learning_rate,
                        self.l2_threshold,
                        self.l2_lambda,
                        tree_privacy_budget,
                        delta_g,
                        delta_v,
                        self.loss,
                        max_depth=self.max_depth,
                        max_leaves=self.max_leaves,
                        min_samples_split=self.min_samples_split,
                        leaf_clipping=self.leaf_clipping,
                        use_bfs=self.use_bfs,
                        use_3_trees=self.use_3_trees,
                        use_decay=self.use_decay,
                        splitting_grid=self.splitting_grid,
                        cat_idx=self.cat_idx,
                        num_idx=self.num_idx,
                    )
                    # in multi-class classification, the target has to be binary
                    # as each tree is a per-class regressor
                    y_target = (
                        (y_tree == kth_tree).astype(np.float64)
                        if self.loss.is_multi_class
                        else y_tree
                    )
                    tree.Fit(X_tree, y_target, gradients_tree)

                    # Add the tree to its corresponding ensemble
                    k_trees.append(tree)
            else:
                # Fit a normal decision tree
                k_trees = []
                for kth_tree in range(self.loss.K):
                    if tree_index == 0:
                        # First tree, start with initial scores (mean of labels)
                        assert self.init_score is not None
                        gradients = self.ComputeGradientForLossFunction(
                            y, self.init_score[: len(y)], kth_tree
                        )
                    else:
                        # Update gradients of all training instances on loss l
                        if update_gradients:
                            gradients = self.ComputeGradientForLossFunction(
                                y_ensemble, self.Predict(X_ensemble), kth_tree
                            )  # type: ignore
                    tree = DifferentiallyPrivateTree(
                        tree_index,
                        self.learning_rate,
                        self.l2_threshold,
                        self.l2_lambda,
                        privacy_budget=0.0,  # Legacy code (instead of None)
                        delta_g=0.0,
                        delta_v=0.0,
                        loss=self.loss,
                        max_depth=self.max_depth,
                        max_leaves=self.max_leaves,
                        min_samples_split=self.min_samples_split,
                        use_bfs=self.use_bfs,
                        use_3_trees=self.use_3_trees,
                        use_decay=self.use_decay,
                        cat_idx=self.cat_idx,
                        num_idx=self.num_idx,
                    )
                    tree.Fit(
                        X,
                        (y == kth_tree).astype(np.float64)
                        if self.loss.is_multi_class
                        else y,
                        gradients,
                    )
                    # Add the tree to its corresponding ensemble
                    k_trees.append(tree)
            self.trees.append(k_trees)

            raw_predictions = self.Predict(X)
            current_loss = self.loss(y, raw_predictions)
            logger.info(
                "#loss_evolution# --- fitting decision tree %d; previous loss: %f; "
                "current loss: %f",
                tree_index,
                prev_loss,
                current_loss,
            )

            new_tree_is_usefull = self.use_new_tree(
                y, raw_predictions, prev_loss, current_loss
            )

            if new_tree_is_usefull:
                logger.info(
                    "#tree_evolution# --- ensemble includes decision tree %d",
                    tree_index,
                )
                update_gradients = True
                prev_loss = current_loss
                # Remove the selected rows from the ensemble's dataset
                # The instances that were filtered out by GBF can still be used for the
                # training of the next trees
                if self.use_dp:
                    X_ensemble = split_data["X_ensemble"]
                    y_ensemble = split_data["y_ensemble"]
                    logger.debug(
                        "Success fitting tree %d on %d instances. Instances left "
                        "for the ensemble: %d",
                        tree_index,
                        len(X_tree),
                        len(X_ensemble),
                    )
                else:
                    early_stop_round = self.early_stop
            else:
                # This tree doesn't improve overall prediction quality, removing from
                # model
                # not reusing gradients in multi-class as they are class-dependent
                logger.info(
                    "#tree_evolution# --- ensemble excludes decision tree %d",
                    tree_index,
                )

                update_gradients = self.loss.is_multi_class
                self.trees.pop()
                if not self.use_dp:
                    early_stop_round -= 1
                    if early_stop_round == 0:
                        logger.info("Early stop kicked in. No improvement, stopping.")
                        break

        return self

    def _gradient_filtering_mask(self, data):
        mask = (self.l2_threshold <= data) & (data <= self.l2_threshold)
        return mask

    def _partition(self, mask, data):
        return (data[mask], data[~mask])

    def _sample(
        self,
        accepted: np.array,
        rejected: np.array,
        n_samples: int,
        post_processing: Optional[Callable[[np.array], np.array]] = None,
        seed: int = None,
    ) -> tuple[np.array, np.array]:
        """Draw `n_samples` samples from `accepted` and `rejected`.

      Prefer `accepted`; draw from `rejected` only if the number of
      items in `accepted` is insufficient. Post-processing may be
      applied on what is drawn from `rejected` (default is no post-
      processing)."""
        if post_processing is None:
            post_processing = lambda x: x

        lk = len(accepted)
        if n_samples <= lk:
            # all good, no need to fill up by rejected
            residual, sample = train_test_split(
                accepted, test_size=n_samples, random_state=seed
            )
            return (sample, np.append(residual, rejected, axis=0))
        else:
            n_rejected = len(rejected)
            assert n_samples - lk <= n_rejected
            if n_samples == n_rejected:
                # No split needed if taking the whole part
                return (rejected, np.array([]))
            # need to fill up by a random selection of rejected
            residual, padding = train_test_split(
                rejected, test_size=n_samples - lk, random_state=seed
            )
            padding = post_processing(padding)
            sample = np.append(accepted, padding, axis=0)
            return (sample, residual)

    def _partition_sampler(
        self,
        mask: np.array,
        n_samples_per_tree: int,
        post_processing: Callable[[np.array], np.array],
        seed: int,
    ) -> Callable[[np.array], tuple[np.array, np.array]]:
        def partition_sample(data):
            accepted, rejected = self._partition(mask, data)
            sample, data = self._sample(
                accepted,
                rejected,
                n_samples_per_tree,
                post_processing=post_processing,
                seed=seed,
            )
            return (sample, data)

        return partition_sample

    def _clip_samples(self, samples):
        return np.clip(samples, -self.l2_threshold, self.l2_threshold)

    def _tree_data_sample_split(
        self,
        gradients: np.array,
        X_ensemble: np.array,
        y_ensemble: np.array,
        n_samples_per_tree: int,
        seed: Optional[int] = None,
    ) -> dict[str, np.array]:

        assert len(gradients) == len(X_ensemble) == len(y_ensemble)

        if self.gradient_filtering and not self.loss.is_multi_class:
            logger.debug("Performing gradient-based data filtering")
            mask = self._gradient_filtering_mask(gradients)
        else:
            # Having a mask of zeros will put all data points into
            # `rejected` and therefore it will be reachable by post-
            # processing, i.e. clipping.
            logger.debug("Not performing gradient-based data filtering")
            mask = np.zeros(len(gradients), dtype=int)

        rng = np.random.default_rng(seed)
        # The upper bound was chosen arbitrarily. It is only important
        # that `worker` uses the same seed within each call of this
        # method, so that the split results do still correspond.
        rand_int = rng.integers(2 ** 32)

        sampler = self._partition_sampler(
            mask, n_samples_per_tree, self._clip_samples, rand_int
        )

        results = [sampler(data) for data in [gradients, X_ensemble, y_ensemble]]
        return dict(
            gradients_tree=results[0][0],
            gradients=results[0][1],
            X_tree=results[1][0],
            X_ensemble=results[1][1],
            y_tree=results[2][0],
            y_ensemble=results[2][1],
        )

    def Predict(self, X: np.array) -> np.array:
        """Predict values from the ensemble of gradient boosted trees.

    See https://github.com/microsoft/LightGBM/issues/1778.

    Args:
      X (np.array): The dataset for which to predict values.

    Returns:
      np.array of shape (n_samples, K): The predictions.
    """
        # sum across the ensemble per class
        tree_predictions = [
            [self.learning_rate * tree.Predict(X) for tree in k_trees]
            for k_trees in self.trees
        ]
        predictions = np.sum(tree_predictions, axis=0).T
        assert self.init_score is not None
        init_score = self.init_score[: len(predictions)]
        return np.add(init_score, predictions)

    def PredictLabels(self, X: np.ndarray) -> np.ndarray:
        """Predict labels out of the raw prediction values of `Predict`.
       Only defined for classification tasks.

    Args:
      X (np.ndarray): The dataset for which to predict labels.

    Returns:
      np.ndarray: The label predictions.

    Raises:
      ValueError: If the loss function doesn't match the prediction task.
    """
        if not hasattr(self.loss, "_raw_prediction_to_decision"):
            raise ValueError("Labels are not defined for regression tasks.")

        raw_predictions = self.Predict(X)
        # pylint: disable=no-member,protected-access
        encoded_labels = self.loss._raw_prediction_to_decision(raw_predictions)
        # pylint: enable=no-member,protected-access
        return encoded_labels

    def PredictProba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X.

    Args:
      X (np.ndarray): The dataset for which to predict labels.

    Returns:
      np.ndarray: The class probabilities of the input samples.

    Raises:
      ValueError: If the loss function doesn't match the prediction task.
    """
        if not hasattr(self.loss, "_raw_prediction_to_proba"):
            raise ValueError("Labels are not defined for regression tasks.")

        raw_predictions = self.Predict(X)
        # pylint: disable=no-member,protected-access
        probas = self.loss._raw_prediction_to_proba(raw_predictions)
        # pylint: enable=no-member,protected-access
        return probas

    def ComputeGradientForLossFunction(
        self, y: np.array, y_pred: np.array, k: int
    ) -> np.array:
        """Compute the gradient of the loss function.

    Args:
      y (np.array): The true values.
      y_pred (np.array): The predictions.
      k (int): the class.

    Returns:
      (np.array): The gradient of the loss function.
    """
        if self.loss.is_multi_class:
            y = (y == k).astype(np.float64)
        # sklearn's impl is using the negative gradient (i.e. y - F).
        # Here the positive gradient is used though
        return -self.loss.negative_gradient(y, y_pred, k=k)


class DecisionNode:
    """Implement a decision node.

  Attributes:
    X (np.array): The dataset.
    y (np.ndarray): The dataset labels.
    gradients (np.array): The gradients for the dataset instances.
    index (int): An index for the feature on which the node splits.
    value (Any): The corresponding value for that index.
    depth (int): The depth of the node.
    left_child (DecisionNode): The left child of the node, if any.
    right_child (DecisionNode): The right child of the node, if any.
    prediction (float): For a leaf node, holds the predicted value.
    processed (bool): If a node has been processed during BFS tree construction.
  """

    # pylint: disable=too-many-arguments

    def __init__(
        self,
        X: Optional[np.array] = None,
        y: Optional[np.array] = None,
        gradients: Optional[np.array] = None,
        index: Optional[int] = None,
        value: Optional[Any] = None,
        gain: Optional[Any] = None,
        depth: Optional[int] = None,
        left_child: Optional["DecisionNode"] = None,
        right_child: Optional["DecisionNode"] = None,
        prediction: Optional[float] = None,
    ) -> None:
        """Initialize a decision node.

    Args:
      X (np.array): Optional. The dataset associated to the node. Only for
          BFS tree and 3-tree construction.
      y (np.ndarray): Optional. The dataset labels associated to the node and
          used for the leaf predictions. Only for BFS tree and 3-tree
          construction.
      gradients (np.array): The gradients for the dataset instances.
      index (int): Optional. An index for the feature on which the node splits.
          Default is None.
      value (Any): Optional. The corresponding value for that index. Default
          is None.
      gain (Any): Optional. The gain that splitting on this value and this
          feature generates. Default is None.
      depth (int): Optional. The depth for the node. Only for BFS tree
          construction.
      left_child (DecisionNode): Optional. The left child of the node, if any.
          Default is None.
      right_child (DecisionNode): Optional. The right child of the node, if any.
          Default is None.
      prediction (float): Optional. For a leaf node, holds the predicted value.
          Default is None.
    """
        # pylint: disable=invalid-name

        self.X = X
        self.y = y
        self.gradients = gradients
        self.index = index
        self.value = value
        self.gain = gain
        self.depth = depth
        self.left_child = left_child
        self.right_child = right_child
        self.prediction = prediction
        self.processed = False

        # To export the tree and plot it, we have to conform to scikit's attributes
        self.n_outputs = 1
        self.node_id = None  # type: Optional[int]


class TreeExporter:
    """Class to export and plot decision trees using Scikit's library."""

    def __init__(self, nodes: List[DecisionNode], l2_lambda: float) -> None:
        self.l2_lambda = l2_lambda
        nodes = sorted(nodes, key=lambda x: x.node_id)
        self.n_outputs = 1
        self.value = []
        self.children_left = []
        self.children_right = []
        self.threshold = []
        self.impurity = []
        for node in nodes:
            if node.prediction is not None:
                self.value.append(np.asarray([[node.prediction]]))
            else:
                # Not a leaf node, so we return that value it'd have been if this was
                # a leaf node
                if node.node_id != 0:  # skipping root node
                    assert node.gradients is not None
                    intermediate_pred = (
                        -1
                        * np.sum(node.gradients)
                        / (len(node.gradients) + self.l2_lambda)
                    )  # type: float
                    self.value.append(np.asarray([[intermediate_pred]]))
                else:
                    self.value.append(np.asarray([[0.0]]))

            if node.value is not None:
                self.threshold.append(node.value)
            else:
                self.threshold.append(-1)

            if not node.left_child:
                self.children_left.append(-1)
            else:
                self.children_left.append(node.left_child.node_id)  # type: ignore
            if not node.right_child:
                self.children_right.append(-1)
            else:
                self.children_right.append(node.right_child.node_id)  # type: ignore

            if node.gain is not None:
                self.impurity.append(node.gain)
            else:
                self.impurity.append(-1)

        self.feature = [node.index for node in nodes]
        self.n_node_samples = [len(node.X) for node in nodes]  # type: ignore
        self.n_classes = [1]
        self.weighted_n_node_samples = np.full(fill_value=1, shape=len(nodes))


class DifferentiallyPrivateTree(BaseEstimator):  # type: ignore
    """Implement a differentially private decision tree.

  Attributes:
    root_node (DecisionNode): The root node of the decision tree.
    nodes_bfs (List[DecisionNode]): All nodes in the tree.
    tree_index (int): The index of the tree being trained.
    learning_rate (float): The learning rate.
    l2_threshold (float): Threshold for leaf clipping.
    l2_lambda (float): Regularization parameter for l2 loss function.
    privacy_budget (float): The tree's privacy budget.
    delta_g (float): The utility function's sensitivity.
    delta_v (float): The sensitivity for leaf clipping.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    max_depth (int): Max. depth for the tree.
  """

    # pylint: disable=invalid-name,too-many-arguments

    def __init__(
        self,
        tree_index: int,
        learning_rate: float,
        l2_threshold: float,
        l2_lambda: float,
        privacy_budget: float,
        delta_g: float,
        delta_v: float,
        loss: LossFunction,
        max_depth: int = 6,
        max_leaves: Optional[int] = None,
        min_samples_split: int = 2,
        leaf_clipping: bool = False,
        use_bfs: bool = False,
        use_3_trees: bool = False,
        use_decay: bool = False,
        splitting_grid: Optional[Sequence[Any]] = None,
        cat_idx: Optional[List[int]] = None,
        num_idx: Optional[List[int]] = None,
    ) -> None:
        """Initialize the decision tree.

    Args:
      tree_index (int): The index of the tree being trained.
      learning_rate (float): The learning rate.
      l2_threshold (float): Threshold for leaf clipping.
      l2_lambda (float): Regularization parameter for l2 loss function.
      privacy_budget (float): The tree's privacy budget.
      delta_g (float): The utility function's sensitivity.
      delta_v (float): The sensitivity for leaf clipping.
      loss (LossFunction): An sklearn loss wrapper
          suitable for regression and classification.
          Valid options are: `LeastSquaresError` or `MultinomialDeviance`.
      max_depth (int): Optional. Max. depth for the tree. Default is 6.
      max_leaves (int): Optional. The max number of leaf nodes for the trees.
          Tree will grow in a best-leaf first fashion until it contains
          max_leaves or until it reaches maximum depth, whichever comes first.
      min_samples_split (int): Optional. The minimum number of samples required
          to split an internal node. Default is 2.
      leaf_clipping (bool): Optional. Whether or not to clip the leaves
          after training. Default is False.
      use_bfs (bool): Optional. If max_leaves is specified, then this is
          automatically True. This will build the tree in a BFS fashion instead
          of DFS. Default is False.
      use_3_trees (bool): Optional. If True, only build trees that have 3
          nodes, and then assemble nb_trees based on these sub-trees, at random.
          Default is False.
      use_decay (bool): Optional. If True, internal node privacy budget has a
          decaying factor.
      cat_idx (List): Optional. List of indices for categorical features.
      num_idx (List): Optional. List of indices for numerical features.
    """

        self.root_node = None  # type: Optional[DecisionNode]
        self.nodes_bfs = Queue()  # type: Queue[DecisionNode]
        self.nodes = []  # type: List[DecisionNode]
        self.tree_index = tree_index
        self.learning_rate = learning_rate
        self.l2_threshold = l2_threshold
        self.l2_lambda = l2_lambda
        # For compatibility reasons, 0.0 means deactivating (instead of None)
        self.privacy_budget = privacy_budget
        self.delta_g = delta_g
        self.delta_v = delta_v
        self.loss = loss
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.leaf_clipping = leaf_clipping
        self.use_bfs = use_bfs
        self.use_3_trees = use_3_trees
        self.use_decay = use_decay
        self.splitting_grid = splitting_grid
        self.cat_idx = cat_idx
        self.num_idx = num_idx

        # Whether to apply DP.
        self.use_dp = privacy_budget > 0.0

        # This handles attribute comparison depending on the attribute's nature
        self.feature_to_op = defaultdict(
            lambda: (operator.lt, operator.ge)
        )  # type: Dict[int, Any]
        if self.cat_idx:
            for feature_index in self.cat_idx:
                self.feature_to_op[feature_index] = (operator.eq, operator.ne)

        if self.max_leaves and not use_bfs:
            # If max_leaves is specified, we grow the tree in a best-leaf first
            # approach
            self.use_bfs = True

        # To keep track of total number of leaves in the tree
        self.current_number_of_leaves = 0
        self.max_leaves_reached = False

        # To export the tree and plot it, we have to conform to scikit's attributes
        self.criterion = "gain"
        self.tree_ = None  # type: Optional[TreeExporter]

    def Fit(self, X: np.array, y: np.ndarray, gradients: np.array) -> None:
        """Fit the tree to the data.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
    """

        # Construct the tree recursively
        if self.use_bfs:
            self.root_node = self.MakeTreeBFS(X, y, gradients)
        else:
            current_depth = 0
            self.root_node = self.MakeTreeDFS(
                X, y, gradients, current_depth, self.max_depth
            )

        leaves = [node for node in self.nodes if node.prediction]

        if self.use_dp:
            if self.leaf_clipping or self.loss.is_multi_class:
                # Clip the leaf nodes
                logger.debug("Performing geometric leaf clipping")
                ClipLeaves(
                    leaves, self.l2_threshold, self.learning_rate, self.tree_index
                )

            # Add noise to the predictions
            privacy_budget_for_leaf_node = self.privacy_budget / 2
            laplace_scale = self.delta_v / privacy_budget_for_leaf_node
            logger.debug("Adding Laplace noise with scale: {0:f}".format(laplace_scale))
            AddLaplacianNoise(leaves, laplace_scale)

        # Make the tree exportable if we want to print it
        # Assign unique IDs to nodes
        node_ids = Queue()  # type: Queue[int]
        for node_id in range(0, len(self.nodes)):
            node_ids.put(node_id)
        if not self.use_bfs:
            self.AssignNodeIDs(self.root_node, node_ids)
        else:
            root_node = max(self.nodes, key=lambda x: len(x.X))  # type: ignore
            self.AssignNodeIDs(root_node, node_ids)
        self.tree_ = TreeExporter(self.nodes, self.l2_lambda)

    def MakeTreeDFS(
        self,
        X: np.array,
        y: np.ndarray,
        gradients: np.array,
        current_depth: int,
        max_depth: int,
        X_sibling: Optional[np.array] = None,
        gradients_sibling: Optional[np.array] = None,
    ) -> DecisionNode:
        """Build a tree recursively, in DFS fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.
      current_depth (int): Current depth for the tree.
      max_depth (int): Max depth for the tree.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      DecisionNode: A decision node.
    """

        def MakeLeafNode() -> DecisionNode:
            node = DecisionNode(
                X=X,
                y=y,
                gradients=gradients,
                prediction=self.GetLeafPrediction(gradients, y),
                depth=current_depth,
            )
            self.nodes.append(node)
            return node

        if current_depth == max_depth or len(X) < self.min_samples_split:
            # Max depth reached or not enough samples to split node, node is a leaf
            # node
            return MakeLeafNode()

        if self.splitting_grid is None:
            _splitting_grid = X.T
        else:
            _splitting_grid = self.splitting_grid

        if not self.use_3_trees:
            best_split = self.FindBestSplit(
                X, gradients, _splitting_grid, current_depth
            )
        else:
            if current_depth != 0:
                best_split = self.FindBestSplit(
                    X,
                    gradients,
                    _splitting_grid,
                    current_depth,
                    X_sibling=X_sibling,
                    gradients_sibling=gradients_sibling,
                )
            else:
                best_split = self.FindBestSplit(
                    X, gradients, _splitting_grid, current_depth
                )
        if best_split:
            logger.debug(
                "Tree DFS: best split found at index {0:d}, value {1:f} "
                "with gain {2:f}. Current depth is {3:d}".format(
                    best_split["index"],
                    best_split["value"],
                    best_split["gain"],
                    current_depth,
                )
            )
            lhs_op, rhs_op = self.feature_to_op[best_split["index"]]
            lhs = np.where(lhs_op(X[:, best_split["index"]], best_split["value"]))[0]
            rhs = np.where(rhs_op(X[:, best_split["index"]], best_split["value"]))[0]
            if not self.use_3_trees:
                left_child = self.MakeTreeDFS(
                    X[lhs], y[lhs], gradients[lhs], current_depth + 1, max_depth
                )
                right_child = self.MakeTreeDFS(
                    X[rhs], y[rhs], gradients[rhs], current_depth + 1, max_depth
                )
            else:
                left_child = self.MakeTreeDFS(
                    X[lhs],
                    y[lhs],
                    gradients[lhs],
                    current_depth + 1,
                    max_depth,
                    X_sibling=X[rhs],
                    gradients_sibling=gradients[rhs],
                )
                right_child = self.MakeTreeDFS(
                    X[rhs],
                    y[rhs],
                    gradients[rhs],
                    current_depth + 1,
                    max_depth,
                    X_sibling=X[lhs],
                    gradients_sibling=gradients[lhs],
                )
            node = DecisionNode(
                X=X,
                gradients=gradients,
                index=best_split["index"],
                value=best_split["value"],
                gain=best_split["gain"],
                left_child=left_child,
                right_child=right_child,
                depth=current_depth,
            )
            self.nodes.append(node)
            return node

        return MakeLeafNode()

    def MakeTreeBFS(
        self, X: np.array, y: np.ndarray, gradients: np.array
    ) -> DecisionNode:
        """Build a tree in a best-leaf first fashion.

    Args:
      X (np.array): The dataset.
      y (np.ndarray): The dataset labels.
      gradients (np.array): The gradients for the dataset instances.

    Returns:
      DecisionNode: A decision node.
    """

        best_split = self.FindBestSplit(X, gradients, current_depth=0)
        if not best_split:
            node = DecisionNode(
                X=X,
                gradients=gradients,
                prediction=self.GetLeafPrediction(gradients, y),
            )
            self.nodes.append(node)
            return node

        logger.debug(
            "Tree BFS: best split found at index {0:d}, value {1:f} with "
            "gain {2:f}.".format(
                best_split["index"], best_split["value"], best_split["gain"]
            )
        )

        # Root node
        node = DecisionNode(
            X=X,
            y=y,
            gradients=gradients,
            index=best_split["index"],
            value=best_split["value"],
            gain=best_split["gain"],
            depth=0,
        )
        self.nodes_bfs.put(node)
        self._ExpandTreeBFS()
        for node in self.nodes:
            # Assigning predictions to remaining leaf nodes if we had to stop
            # constructing the tree early because we reached max number of leaf nodes
            if not node.prediction and not node.left_child and not node.right_child:
                node.prediction = self.GetLeafPrediction(node.gradients, node.y)
        return node

    def _ExpandTreeBFS(self) -> None:
        """Expand a tree in a best-leaf first fashion.

    Implement https://researchcommons.waikato.ac.nz/bitstream/handle/10289/2317
    /thesis.pdf?sequence=1&isAllowed=y
    """

        # Node queue is empty or too many leaves, stopping
        if self.nodes_bfs.empty() or self.max_leaves_reached:
            return None

        current_node = self.nodes_bfs.get()

        # If there are not enough samples to split in that node, make it a leaf
        # node and process next node
        assert current_node.gradients is not None
        if len(current_node.gradients) < self.min_samples_split:
            self._MakeLeaf(current_node)
            if not self._IsMaxLeafReached():
                return self._ExpandTreeBFS()
            return None

        # If we reached max depth
        if current_node.depth == self.max_depth:
            self._MakeLeaf(current_node)
            if not self._IsMaxLeafReached():
                if self.max_leaves:
                    return self._ExpandTreeBFS()
                while not self.nodes_bfs.empty():
                    node = self.nodes_bfs.get()
                    self._MakeLeaf(node)
            return None

        #  Do the split
        assert current_node.X is not None
        assert current_node.y is not None
        assert current_node.gradients is not None
        assert current_node.depth is not None
        lhs_op, rhs_op = self.feature_to_op[current_node.index]  # type: ignore
        lhs = np.where(
            lhs_op(current_node.X[:, current_node.index], current_node.value)
        )[0]
        rhs = np.where(
            rhs_op(current_node.X[:, current_node.index], current_node.value)
        )[0]
        lhs_X, rhs_X = current_node.X[lhs], current_node.X[rhs]
        lhs_grad, rhs_grad = current_node.gradients[lhs], current_node.gradients[rhs]

        lhs_y, rhs_y = current_node.y[lhs], current_node.y[rhs]
        lhs_best_split = self.FindBestSplit(
            lhs_X, lhs_grad, current_depth=current_node.depth + 1
        )
        rhs_best_split = self.FindBestSplit(
            rhs_X, rhs_grad, current_depth=current_node.depth + 1
        )

        # Can't split the node, so this becomes a leaf node.
        if not lhs_best_split or not rhs_best_split:
            self._MakeLeaf(current_node)
            if not self._IsMaxLeafReached():
                return self._ExpandTreeBFS()
            return None

        logger.debug(
            "Tree BFS: best split found at index {0:d}, value {1:f} with "
            "gain {2:f}.".format(
                lhs_best_split["index"], lhs_best_split["value"], lhs_best_split["gain"]
            )
        )
        logger.debug(
            "Tree BFS: best split found at index {0:d}, value {1:f} with "
            "gain {2:f}.".format(
                rhs_best_split["index"], rhs_best_split["value"], rhs_best_split["gain"]
            )
        )

        # Splitting the node is possible, creating the children
        assert current_node.depth is not None
        left_child = DecisionNode(
            X=lhs_X,
            y=lhs_y,
            gradients=lhs_grad,
            index=lhs_best_split["index"],
            value=lhs_best_split["value"],
            gain=lhs_best_split["gain"],
            depth=current_node.depth + 1,
        )
        right_child = DecisionNode(
            X=rhs_X,
            y=rhs_y,
            gradients=rhs_grad,
            index=rhs_best_split["index"],
            value=rhs_best_split["value"],
            gain=rhs_best_split["gain"],
            depth=current_node.depth + 1,
        )

        current_node.left_child = left_child
        current_node.right_child = right_child
        self.nodes.append(current_node)

        # Adding them to the list of nodes for further expansion in best-gain order
        if lhs_best_split["gain"] >= rhs_best_split["gain"]:
            self.nodes_bfs.put(left_child)
            self.nodes_bfs.put(right_child)
        else:
            self.nodes_bfs.put(right_child)
            self.nodes_bfs.put(left_child)
        return self._ExpandTreeBFS()

    def _MakeLeaf(self, node: DecisionNode) -> None:
        """Make a node a leaf node.

    Args:
      node (DecisionNode): The node to make a leaf from.
    """
        node.prediction = self.GetLeafPrediction(node.gradients, node.y)
        self.current_number_of_leaves += 1
        self.nodes.append(node)

    def _IsMaxLeafReached(self) -> bool:
        """Check if we reached maximum number of leaf nodes.

    Returns:
      bool: True if we reached the maximum number of leaf nodes,
          False otherwise.
    """
        leaf_candidates = 0
        for node in list(self.nodes_bfs.queue):
            if not node.left_child and not node.right_child:
                leaf_candidates += 1
        if self.max_leaves:
            if self.current_number_of_leaves + leaf_candidates >= self.max_leaves:
                self.max_leaves_reached = True
        return self.max_leaves_reached

    def FindBestSplit(
        self,
        X: np.array,
        gradients: np.array,
        featurewise_split_candidates: Sequence[np.array],
        current_depth: Optional[int] = None,
        X_sibling: Optional[np.array] = None,
        gradients_sibling: Optional[np.array] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find best split of data using the exponential mechanism.

    Args:
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.
      featurewise_split_candidates (Sequence[np.array]):
          Per feature index an array of split candidates. It is assumed,
          but not checked, that each array contains no duplicates.
      current_depth (int): Optional. The current depth of the tree. If
          specified, the privacy budget decays with the depth growing.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      Optional[Dict[str, Any]]: A dictionary containing the split
          information, or none if no split could be done.
    """

        if self.use_dp:
            if current_depth and self.use_decay:
                privacy_budget_for_node = np.around(
                    np.divide(self.privacy_budget / 2, np.power(2, current_depth)),
                    decimals=7,
                )
            else:
                privacy_budget_for_node = np.around(
                    np.divide(self.privacy_budget / 2, self.max_depth), decimals=7
                )

            if self.use_3_trees and current_depth != 0:
                # If not for the root node splitting, budget is divided by the 3-nodes
                privacy_budget_for_node /= 2

            logger.debug(
                "Using {0:f} budget for internal leaf nodes.".format(
                    privacy_budget_for_node
                )
            )

        probabilities = []
        max_gain = -np.inf
        # Iterate over features
        for feature_idx, split_cands in enumerate(featurewise_split_candidates):
            binary_split = len(split_cands) == 2
            # Iterate over unique value for this feature
            for idx, candidate in enumerate(split_cands):
                # Find gain for that split
                if binary_split and idx == 1:
                    # If the attribute only has 2 values then we don't need to care for
                    # both gains as they're equal
                    prob = {"index": feature_idx, "value": candidate, "gain": 0.0}
                else:
                    gain = self.ComputeGain(
                        feature_idx,
                        candidate,
                        X,
                        gradients,
                        X_sibling=X_sibling,
                        gradients_sibling=gradients_sibling,
                    )
                    if gain == -1:
                        # Feature's value cannot be chosen, skipping
                        continue
                    # Compute probability for exponential mechanism
                    if self.use_dp:
                        gain = (privacy_budget_for_node * gain) / (2.0 * self.delta_g)
                    if gain > max_gain:
                        max_gain = gain
                    prob = {"index": feature_idx, "value": candidate, "gain": gain}
                probabilities.append(prob)
        if self.use_dp:
            return ExponentialMechanism(probabilities, max_gain)
        return max(probabilities, key=lambda x: x["gain"]) if probabilities else None

    def GetLeafPrediction(self, gradients: np.array, y: np.ndarray) -> float:
        """Compute the leaf prediction.

    Args:
      gradients (np.array): The gradients for the dataset instances.
      y (np.ndarray): The dataset labels.

    Returns:
      float: The prediction for the leaf node
    """
        return ComputePredictions(gradients, y, self.loss, self.l2_lambda)

    def Predict(self, X: np.array) -> np.array:
        """Return predictions for a list of input data.

    Args:
      X: The input data used for prediction.

    Returns:
      np.array: An array with the predictions.
    """
        predictions = []
        for row in X:
            predictions.append(self._Predict(row, self.root_node))  # type: ignore
        return np.asarray(predictions)

    def _Predict(self, row: np.array, node: DecisionNode) -> float:
        """Walk through the decision tree to output a prediction for the row.

    Args:
      row (np.array): The row to classify.
      node (DecisionNode): The current decision node.

    Returns:
      float: A prediction for the row.
    """
        if node.prediction is not None:
            return node.prediction
        assert node.index is not None
        value = row[node.index]
        _, rhs_op = self.feature_to_op[node.index]
        if rhs_op(value, node.value):
            child_node = node.right_child
        else:
            child_node = node.left_child
        return self._Predict(row, child_node)  # type: ignore

    def ComputeGain(
        self,
        index: int,
        value: Any,
        X: np.array,
        gradients: np.array,
        X_sibling: Optional[np.array] = None,
        gradients_sibling: Optional[np.array] = None,
    ) -> float:
        """Compute the gain for a given split.

    See https://dl.acm.org/doi/pdf/10.1145/2939672.2939785

    Args:
      index (int): The index for the feature to split on.
      value (Any): The feature's value to split on.
      X (np.array): The dataset.
      gradients (np.array): The gradients for the dataset instances.
      X_sibling (np.array): Optional. The subset of data in the sibling node.
          For 3_trees only.
      gradients_sibling (np.array): Optional. The gradients in the sibling
          node. For 3_trees only.

    Returns:
      float: The gain for the split.
    """
        lhs_op, rhs_op = self.feature_to_op[index]
        lhs = np.where(lhs_op(X[:, index], value))[0]
        rhs = np.where(rhs_op(X[:, index], value))[0]
        if len(lhs) == 0 or len(rhs) == 0:
            # Can't split on this feature as all instances share the same value
            return -1

        if self.use_3_trees and X_sibling is not None:
            X = np.concatenate((X, X_sibling), axis=0)
            gradients = np.concatenate((gradients, gradients_sibling), axis=0)
            lhs = np.where(lhs_op(X[:, index], value))[0]
            rhs = np.where(rhs_op(X[:, index], value))[0]

        lhs_grad, rhs_grad = gradients[lhs], gradients[rhs]
        lhs_gain = np.square(np.sum(lhs_grad)) / (
            len(lhs) + self.l2_lambda
        )  # type: float
        rhs_gain = np.square(np.sum(rhs_grad)) / (
            len(rhs) + self.l2_lambda
        )  # type: float
        total_gain = lhs_gain + rhs_gain
        return total_gain if total_gain >= 0.0 else 0.0

    def AssignNodeIDs(
        self, node: DecisionNode, node_ids: Queue  # type: ignore
    ) -> None:
        """Walk through the tree and assign a unique ID to the decision nodes.

    Args:
      node (DecisionNode): The node of the tree to assign an ID to.
      node_ids (Queue): Queue that contains all available node ids.
    """
        node.node_id = node_ids.get()
        if node.left_child:
            self.AssignNodeIDs(node.left_child, node_ids)
        if node.right_child:
            self.AssignNodeIDs(node.right_child, node_ids)

    @staticmethod
    def fit() -> None:
        """Stub for BaseEstimator."""
        return

    @staticmethod
    def predict() -> None:
        """Stub for BaseEstimator"""
        return


def ClipLeaves(
    leaves: List[DecisionNode],
    l2_threshold: float,
    learning_rate: float,
    tree_index: int,
) -> None:
    """Clip leaf nodes.

  If the prediction is higher than the threshold, set the prediction to
  that threshold.

  Args:
    leaves (List[DecisionNode]): The leaf nodes.
    l2_threshold (float): Threshold of the l2 loss function.
    learning_rate (float): The learning rate.
    tree_index (int): The index for the current tree.
  """
    threshold = l2_threshold * math.pow((1 - learning_rate), tree_index)
    for leaf in leaves:
        assert leaf.prediction is not None
        if np.abs(leaf.prediction) > threshold:
            if leaf.prediction > 0:
                leaf.prediction = threshold
            else:
                leaf.prediction = -1 * threshold


def AddLaplacianNoise(leaves: List[DecisionNode], scale: float) -> None:
    """Add laplacian noise to the leaf nodes.

  Args:
    leaves (List[DecisionNode]): The list of leaves.
    scale (float): The scale to use for the laplacian distribution.
  """

    for leaf in leaves:
        noise = np.random.laplace(0, scale)
        logger.debug("Leaf value before noise: {0:f}".format(np.float(leaf.prediction)))
        leaf.prediction += noise
        logger.debug("Leaf value after noise: {0:f}".format(np.float(leaf.prediction)))


def ComputePredictions(
    gradients: np.ndarray, y: np.ndarray, loss: LossFunction, l2_lambda: float
) -> float:
    """Computes the predictions of a leaf.

  Used in the `DifferentiallyPrivateTree` as well as in `SplitNode`
  for the 3-tree version.

  Ref:
    Friedman 01. "Greedy function approximation: A gradient boosting machine."
      (https://projecteuclid.org/euclid.aos/1013203451)

  Args:
    gradients (np.ndarray): The positive gradients y˜ for the dataset instances.
    y (np.ndarray): The dataset labels y.
    loss (LossFunction): An sklearn loss wrapper
        suitable for regression and classification.
    l2_lambda (float): Regularization parameter for l2 loss function.

  Returns:
    Prediction γ of a leaf
  """
    if len(gradients) == 0:
        prediction = 0.0  # type: ignore
    elif loss.is_multi_class:
        # sum of neg. gradients divided by sum of 2nd derivatives
        # aka one Newton-Raphson step
        # for details ref. (eq 33+34) in Friedman 01.
        prediction = -1 * np.sum(gradients) * (loss.K - 1) / loss.K
        denom = np.sum((y + gradients) * (1 - y - gradients))
        prediction = 0 if abs(denom) < 1e-150 else prediction / (denom + l2_lambda)
    else:
        prediction = -1 * np.sum(gradients) / (len(gradients) + l2_lambda)
    return prediction


def ExponentialMechanism(
    probabilities: List[Dict[str, Any]], max_gain: float, reverse: bool = False
) -> Optional[Dict[str, Any]]:
    """Apply the exponential mechanism.

  Args:
    probabilities (List[Dict]): List of probabilities to choose from.
    max_gain (float): The maximum gain amongst all probabilities in the list.
    reverse (bool): Optional. If True, sort probabilities in reverse order (
        i.e. lower gains are better).

  Returns:
    Dict: a candidate (i.e. probability) from the list.
  """

    if (np.asarray([prob["gain"] for prob in probabilities]) <= 0.0).all():
        # No split offers a positive gain, node should be a leaf node
        return None

    with np.errstate(all="raise"):
        try:
            gains = np.asarray(
                [prob["gain"] for prob in probabilities if prob["gain"] != 0.0],
                dtype=np.float128,
            )
            for prob in probabilities:
                # e^0 is 1, so checking for that
                if prob["gain"] <= 0.0:
                    prob["probability"] = 0.0
                else:
                    prob["probability"] = np.exp(prob["gain"] - logsumexp(gains))
        # Happens when np.exp() overflows because of a gain that's too high
        except FloatingPointError:
            for prob in probabilities:
                gain = prob["gain"]
                if gain > 0.0:
                    # Check if the gain of each candidate is too small compared to
                    # the max gain seen up until now. If so, set the probability for
                    # this split to 0.
                    try:
                        _ = np.exp(max_gain - gain)
                    except FloatingPointError:
                        prob["probability"] = 0.0
                    # If it's not too small, we need to compute a new sum that
                    # doesn't overflow. For that we only take into account 'large'
                    # gains with respect to the current candidate. If again the
                    # difference is so small that it would still overflow, we set the
                    # probability for this split to 0.
                    sub_sum_exp = 0.0
                    try:
                        sub_sum_exp = logsumexp(
                            np.asarray(gains - gain, dtype=np.float128)
                        )
                    except FloatingPointError:
                        prob["probability"] = 0.0

                    # Other candidates compare similarly, so we can compute a
                    # probability. If it underflows, set it to 0 as well.
                    if sub_sum_exp > 0.0:
                        try:
                            prob["probability"] = np.exp(
                                0.0 - sub_sum_exp
                            )  # E.q. to 1/e^sub
                        except FloatingPointError:
                            prob["probability"] = 0.0
                else:
                    prob["probability"] = 0.0

    # Apply the exponential mechanism
    previous_prob = 0.0
    random_prob = np.random.uniform()
    for prob in probabilities:
        if prob["probability"] != 0.0:
            prob["probability"] += previous_prob
            previous_prob = prob["probability"]

    op = operator.ge if not reverse else operator.le
    #  Try and find a candidate at least 10 times before giving up and making
    # the node a leaf node
    for _ in range(10):
        for prob in probabilities:
            if op(prob["probability"], random_prob):
                return prob
        random_prob = np.random.uniform()
    return None
