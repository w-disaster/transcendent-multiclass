import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from termcolor import cprint
from tqdm import tqdm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import transcend.calibration as calibration
import transcend.data as data
import transcend.scores as scores
import transcend.thresholding as thresholding
import transcend.utils as utils
from dataset.malware_dataset import MalwareDataset
from transcend.classifiers.random_forest import RandomForestNCMClassifier


def main():
    # ---------------------------------------- #
    # 0. Prelude                               #
    # ---------------------------------------- #

    utils.configure_logger()
    args = utils.parse_args()

    print("This is ICE - only compute test pvals with cal ncms")

    base_path = "/home/luca/ml-malware-concept-drift/src/notebooks/"

    # Load Full Dataset with Malware Static Features
    malware_dataset = MalwareDataset(
        split=pd.Timestamp("2021-09-03 13:47:49"),
        truncated_fam_path="truncated_samples_per_family.csv",
        truncated_threshold=7,
    )

    logging.info("Loading malw-static-features training features...")
    with open(
            base_path + "clustering/1_preprocessing/X_nontrunc_norm.pickle", "rb"
    ) as f:
        X = pickle.load(f)

    y_train = malware_dataset.training_dataset["family"]

    train_uniq_families, num_samples = np.unique(y_train, return_counts=True)
    families_to_consider = train_uniq_families[(num_samples >= 30) & (num_samples <= 70)]

    logging.info(f"Family drift is analyzed only for {len(families_to_consider)} families")

    full_dataset = malware_dataset.df_malware_family_fsd
    shas = full_dataset[full_dataset["family"].isin(families_to_consider)]["sha256"]

    training_dataset = malware_dataset.training_dataset.set_index("sha256")
    training_dataset = training_dataset.loc[training_dataset.index[training_dataset.index.isin(shas)]]

    testing_dataset = malware_dataset.testing_dataset.set_index("sha256")
    testing_dataset = testing_dataset.loc[testing_dataset.index[testing_dataset.index.isin(shas)]]

    X_train, X_test = X.loc[training_dataset.index], X.loc[testing_dataset.index]
    y_train, y_test = training_dataset["family"], testing_dataset["family"]

    # del X

    # Convert family labels to integers
    all_labels = pd.concat([y_train, y_test]).unique()
    y_train = pd.Categorical(y_train, categories=all_labels).codes
    y_test = pd.Categorical(y_test, categories=all_labels).codes

    logging.info("Loaded: {}".format(X_train.shape, y_train.shape))

    test_size = 0.34

    # saved_data_folder = os.path.join('models', '{}-fold'.format(args.folds))
    saved_data_folder = os.path.join(
        "models", "ice-{}-{}".format(args.folds, "malw-static-features")
    )

    # ---------------------------------------- #
    # 1. Calibration                           #
    # ---------------------------------------- #

    # logging.info("Training calibration set...")
    #
    # X_proper_train, X_cal, y_proper_train, y_cal = train_test_split(
    #     X_train, y_train, test_size=test_size, random_state=3
    # )

    # cal_results_dict = calibration.train_calibration_ice(
    #     X_proper_train=X_proper_train,
    #     X_cal=X_cal,
    #     y_proper_train=y_proper_train,
    #     y_cal=y_cal,
    #     fold_index="ice_{}".format(test_size),
    #     saved_data_folder=saved_data_folder,
    # )
    #
    # cal_results_name = os.path.join(saved_data_folder, 'cal_results.p')
    # data.cache_data(cal_results_dict, cal_results_name)

    # fold_results_list = calibration.train_calibration_set(
    #     X_train, y_train, args.folds, args.ncpu, saved_data_folder)
    # logging.info('Concatenating calibration fold results...')
    # cal_results_dict = calibration.concatenate_calibration_set_results(
    #     fold_results_list)

    # ---------------------------------------- #
    # 3. Generate 'Full' Model for Deployment  #
    # ---------------------------------------- #

    logging.info("Beginning TEST phase.")

    logging.info("Training model on full training set...")

    model_name = "rf_full_train_ice.p"
    # ???
    # model_name = "rf_cal_fold_ice_{}.p".format(test_size)
    model_name = os.path.join(saved_data_folder, model_name)

    rf = RandomForestNCMClassifier()
    rf.fit(X_train, y_train)
    data.cache_data(rf, model_name)

    # ---------------------------------------- #
    # 4. Score and Predict Test Observations   #
    # ---------------------------------------- #

    # P-value scores

    # logging.info("Computing p-values for test ...")
    # y_test_pred = rf.predict(X_test)
    #
    # saved_data_name = "p_vals_ncms_{}_rf_full_test_phase.p".format(
    #     args.pval_consider.replace("-", "_")
    # )
    # saved_data_name = os.path.join(saved_data_folder, saved_data_name)
    #
    # # if True:
    # #     if args.pval_consider == "full-train":
    #
    # logging.info("Getting NCMs for train")
    # ncms_train = scores.get_rf_ncms(rf, X_train, y_train)
    #
    # logging.info("Getting NCMs for test")
    # ncms_test = scores.get_rf_ncms(rf, X_test, y_test_pred)
    #
    # p_val_test_dict = scores.compute_p_values_cred_and_conf(
    #     train_ncms=ncms_train,
    #     groundtruth_train=y_train,
    #     test_ncms=ncms_test,
    #     y_test=y_test_pred,
    # )
    # data.cache_data(p_val_test_dict, saved_data_name)

    # Test appearing families

    all_y_test = malware_dataset.testing_dataset["family"].unique()
    all_y_train = malware_dataset.training_dataset["family"].unique()

    result = np.setdiff1d(all_y_test, all_y_train)

    # y_test_app = pd.Categorical(result, categories=result).codes
    # y_test_app = [int(l) + len(all_labels) for l in y_test_app]
    # y_test_app

    shas_test_app_fam = malware_dataset.testing_dataset[malware_dataset.testing_dataset["family"].isin(result)][
        "sha256"]
    X_test = X.loc[shas_test_app_fam]
    y_test_app_fam_pred = rf.predict(X_test)

    del X

    saved_data_name = "p_vals_ncms_{}_rf_test_phase_app_fam.p".format(
        args.pval_consider.replace("-", "_")
    )
    saved_data_name = os.path.join(saved_data_folder, saved_data_name)

    logging.info("Getting NCMs for train")
    ncms_train = scores.get_rf_ncms(rf, X_train, y_train)

    logging.info("Getting NCMs for test samples of appearing families")
    ncms_test = scores.get_rf_ncms(rf, X_test, y_test_app_fam_pred)

    p_val_test_dict = scores.compute_p_values_cred_and_conf(
        train_ncms=ncms_train,
        groundtruth_train=y_train,
        test_ncms=ncms_test,
        y_test=y_test_app_fam_pred,
    )
    data.cache_data(p_val_test_dict, saved_data_name)


def package_cred_conf(cred_values, conf_values, criteria):
    package = {}

    if "cred" in criteria:
        package["cred"] = cred_values
    if "conf" in criteria:
        package["conf"] = conf_values

    return package


def print_thresholds(binary_thresholds):
    # Display per-class thresholds
    if "cred" in binary_thresholds:
        s = "Cred thresholds: mw {:.6f}, gw {:.6f}".format(
            binary_thresholds["cred"]["mw"], binary_thresholds["cred"]["gw"]
        )
    if "conf" in binary_thresholds:
        s = "Conf thresholds: mw {:.6f}, gw {:.6f}".format(
            binary_thresholds["conf"]["mw"], binary_thresholds["conf"]["gw"]
        )
    logging.info(s)
    return s


if __name__ == "__main__":
    main()
