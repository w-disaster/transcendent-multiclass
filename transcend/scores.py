# -*- coding: utf-8 -*-
from multiprocessing import Pool
from functools import partial
import time
from p_tqdm import p_map

"""
scores.py
~~~~~~~~~

Functions for producing the various scores used during conformal evaluation,
such as non-conformity measures, credibility and confidence p-values and
probabilities for comparison.

Note that the functions in this module currently only apply to producing
scores for a binary classification task and an SVM classifier. Different
settings and different classifiers will require their own functions for
generating non-conformity measures based on different intuitions.

"""

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, shared_memory


def get_rf_ncms(clf, X_in, y_in):
    return clf.ncm(X_in, y_in)


def get_svm_ncms(clf, X_in, y_in):
    """Helper functions to get NCMs across an entire pair of X,y arrays."""
    assert hasattr(clf, "decision_function")
    return [
        get_single_svm_ncm(clf, x, y)
        for x, y in tqdm(zip(X_in, y_in), total=len(y_in), desc="svm ncms")
    ]


def get_single_svm_ncm(clf, single_x, single_y):
    """Collect a non-conformity measure from the classifier for `single_x`.

    A note about SVM ncms: In binary classification with a linear SVM, the
    output score is the distance from the hyperplane with respect to the
    positive class. If the score is negative, the prediction is class 0, if
    positive, it's class 1 (in sklearn technically it will be clf.class_[0] and
    clf.class_[1] respectively). To perform thresholding with conformal
    evaluator, we need the distance from the hyperplane with respect to *both*
    classes, so we simply flip the sign to get the 'reflection' for the other
    class.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the NCMs.
        single_x (np.ndarray): An single feature vector to get the NCM for.
        single_y (int): Either the ground truth (calibration) or predicted
            label (testing) of `single_x`.

    Returns:
        float: The NCM for the given `single_x`.

    """
    assert hasattr(clf, "decision_function")
    decision = clf.decision_function(single_x)

    # If y (ground truth in calibration, prediction in testing) is malware
    # then flip the sign to ensure the most conforming point is most minimal.
    if single_y == 1:
        return -decision
    elif single_y == 0:
        return decision
    raise Exception("Unknown class? Only binary decisions supported.")


def compute_p_values_cred_and_conf(clf, train_ncms, groundtruth_train, test_ncms, y_test, X_test):
    return {
        "conf": compute_confidence_scores(clf, train_ncms, groundtruth_train, y_test, X_test),
        "cred": compute_credibility_scores(train_ncms, groundtruth_train, test_ncms, y_test)
    }


def creds(args):
    i, n, m, op_y_labels = args

    existing_shm_train_ncms = shared_memory.SharedMemory(name=train_ncms_shm_name)
    train_ncms = np.ndarray(train_ncms_len, dtype=train_ncms_dtype, buffer=existing_shm_train_ncms.buf)

    existing_shm_groundtruth_train = shared_memory.SharedMemory(name=groundtruth_train_shm_name)
    groundtruth_train = np.ndarray(groundtruth_train_len, dtype=groundtruth_train_dtype,
                                   buffer=existing_shm_groundtruth_train.buf)

    existing_shm_ncms = shared_memory.SharedMemory(name=ncms_shm_name)
    ncms_X_test_op_y_label = np.ndarray(ncms_shm_len, dtype=ncms_shm_dtype, buffer=existing_shm_ncms.buf)

    def partial_f(single_ncm_test, single_y_test):
        return compute_single_cred_p_value(train_ncms=train_ncms,
                                           groundtruth_train=groundtruth_train,
                                           single_y_test=single_y_test,
                                           single_test_ncm=single_ncm_test)

    return 1 - max([partial_f(single_ncm_test=ncm, single_y_test=op_y) for ncm, op_y in
                    zip([ncms_X_test_op_y_label[i + k * m] for k in range(n)], op_y_labels)])


def compute_credibility_scores(train_ncms, groundtruth_train, test_ncms, y_test):
    return [
        compute_single_cred_p_value(
            train_ncms=train_ncms,
            groundtruth_train=groundtruth_train,
            single_test_ncm=ncm,
            single_y_test=y,
        )
        for ncm, y in tqdm(zip(test_ncms, y_test), total=len(y_test), desc="cred pvals")
    ]


def compute_confidence_scores(clf, train_ncms, groundtruth_train, y_test, X_test):
    unique_y_test_labels = np.unique(y_test)
    y_test_series = pd.Series(y_test, index=X_test.index)

    # Allocate shared memory for multiprocessing

    train_ncms_shm = shared_memory.SharedMemory(create=True, size=train_ncms.nbytes)
    train_ncms_shm_arr = np.ndarray(train_ncms.shape, dtype=train_ncms.dtype, buffer=train_ncms_shm.buf)
    np.copyto(train_ncms_shm_arr, train_ncms)
    global train_ncms_shm_name, train_ncms_len, train_ncms_dtype
    train_ncms_shm_name, train_ncms_len, train_ncms_dtype = train_ncms_shm.name, groundtruth_train.shape, train_ncms.dtype

    groundtruth_train_shm = shared_memory.SharedMemory(create=True, size=groundtruth_train.nbytes)
    groundtruth_train_shm_arr = np.ndarray(groundtruth_train.shape, dtype=groundtruth_train.dtype,
                                           buffer=groundtruth_train_shm.buf)
    np.copyto(groundtruth_train_shm_arr, groundtruth_train)
    global groundtruth_train_shm_name, groundtruth_train_len, groundtruth_train_dtype
    groundtruth_train_shm_name, groundtruth_train_len, groundtruth_train_dtype = (
        groundtruth_train_shm.name, groundtruth_train.shape,
        groundtruth_train.dtype)

    conf_X_test = []
    for y in tqdm(unique_y_test_labels, total=len(unique_y_test_labels), desc="label conf"):
        op_y_labels = unique_y_test_labels[unique_y_test_labels != y]
        X_test_y = X_test[y_test_series == y]

        m = X_test_y.shape[0]
        n = len(op_y_labels)

        M = pd.concat([X_test_y] * n, ignore_index=True)
        Y = np.concat([np.full(m, op_y_label) for op_y_label in op_y_labels])

        t_start = time.time()
        ncms_X_test_op_y_label = clf.ncm(M, Y)
        logging.info(f"TIME RF: {time.time() - t_start}")

        ncms_shm = shared_memory.SharedMemory(create=True, size=ncms_X_test_op_y_label.nbytes)
        ncms_shm_arr = np.ndarray(ncms_X_test_op_y_label.shape, dtype=ncms_X_test_op_y_label.dtype,
                                  buffer=ncms_shm.buf)
        np.copyto(ncms_shm_arr, ncms_X_test_op_y_label)
        global ncms_shm_name, ncms_shm_len, ncms_shm_dtype
        ncms_shm_name, ncms_shm_len, ncms_shm_dtype = (ncms_shm.name, ncms_X_test_op_y_label.shape,
                                                       ncms_X_test_op_y_label.dtype)

        t_start = time.time()
        with Pool(32) as p:
            conf_X_test_y = p.map(creds, [(i, n, m, op_y_labels) for i in range(m)])

        logging.info(f"TIME POOL: {time.time() - t_start}")
        ncms_shm.close()
        ncms_shm.unlink()

        conf_X_test_y = pd.DataFrame(conf_X_test_y, index=X_test_y.index)
        conf_X_test.append(conf_X_test_y)

    train_ncms_shm.close()
    train_ncms_shm.unlink()
    groundtruth_train_shm.close()
    groundtruth_train_shm.unlink()
    _, conf_X_test = X_test.align(pd.concat(conf_X_test, axis=0), axis=0)
    return conf_X_test


def compute_single_cred_p_value(
        train_ncms, groundtruth_train, single_test_ncm, single_y_test
):
    """Compute a single credibility p-value.

    Credibility p-values describe how 'conformal' a point is with respect to
    the other objects of that class. They're computed as the proportion of
    points with greater NCMs (the number of points _less conforming_ than the
    reference point) over the total number of points.

    Intuitively, a point predicted as malware which is the further away from
    the decision boundary than any other point will have the highest p-value
    out of all other malware points. It will have the smallest NCM (as it is
    the least _non-conforming_) and thus no other points will have a greater
    NCM and it will have a credibility p-value of 1.
    """

    mask = groundtruth_train == single_y_test
    single_cred_p_value = sum((train_ncms >= single_test_ncm) & (mask)) / sum(mask)
    return single_cred_p_value


def get_rf_probs(clf, X_in):
    assert hasattr(clf, "predict_proba")
    logging.info(f"SHAPE RF: {X_in.shape}")
    probability_results = clf.predict_proba(X_in)
    probas_cal_fold = [np.max(t) for t in probability_results]
    pred_proba_cal_fold = [np.argmax(t) for t in probability_results]
    return probas_cal_fold, pred_proba_cal_fold


def get_svm_probs(clf, X_in):
    """Get scores and predictions for comparison with probabilities.

    Note that this function returns the predictions _and_ probabilities given
    by the classifier and that these predictions may different from other
    outputs from the same classifier (such as `predict` or `decision_function`.
    This is due to Platt's scaling (and it's implementation in scikit-learn) in
    which a 5-fold SVM is trained and used to score the observation
    (`predict_proba()` is actually the average of these 5 classifiers).

    The takeaway is to be sure that you're always using probability scores with
    probability predictions and not with the output of other SVC functions.

    Args:
        clf (sklearn.svm.SVC): The classifier to use for the probabilities.
        X_in (np.ndarray): An array of feature vectors to classify.

    Returns:
        (list, list): (Probability scores, probability labels) for `X_in`.

    """
    assert hasattr(clf, "predict_proba")
    probability_results = clf.predict_proba(X_in)
    probas_cal_fold = [np.max(t) for t in probability_results]
    pred_proba_cal_fold = [np.argmax(t) for t in probability_results]
    return probas_cal_fold, pred_proba_cal_fold
