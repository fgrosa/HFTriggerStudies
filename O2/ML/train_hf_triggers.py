"""
Script for the training of ML models to be used in HF triggers
"""

import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml

from hipe4ml import plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler


def get_list_input_files(indir, channel):
    """
    function that returns the list of files

    Parameters
    -----------------
    - indir: input directory
    - channel: decay channel, options:
        D0ToKPi, DPlusToPiKPi, DsToKKPi, LcToPKPi, XicToPKPi

    Outputs
    -----------------
    - list_prompt: list of files for prompt
    - list_nonprompt: list of files for nonprompt
    - list_bkg: list of files for bkg
    """

    if channel not in ["D0ToKPi", "DplusToPiKPi", "DsToKKPi", "LcToPKPi", "XicToPKPi"]:
        print(f"ERROR: channel {channel} not implemented, return None")
        return None, None, None

    list_prompt, list_nonprompt, list_bkg = [], [], []
    subdirs = os.listdir(indir)
    for subdir in subdirs:
        subdir = os.path.join(indir, subdir)
        if os.path.isdir(subdir):
            for subsubdir in os.listdir(subdir):
                subsubdir = os.path.join(subdir, subsubdir)
                file_prompt = os.path.join(subsubdir, f"Prompt_{channel}.parquet.gzip")
                file_nonprompt = os.path.join(subsubdir, f"Nonprompt_{channel}.parquet.gzip")
                file_bkg = os.path.join(subsubdir, f"Bkg_{channel}.parquet.gzip")
                if os.path.isfile(file_prompt):
                    list_prompt.append(file_prompt)
                if os.path.isfile(file_nonprompt):
                    list_nonprompt.append(file_nonprompt)
                if os.path.isfile(file_bkg):
                    list_bkg.append(file_bkg)

    return list_prompt, list_nonprompt, list_bkg


def data_prep(config): #pylint: disable=too-many-statements, too-many-branches, too-many-locals
    """
    function for data preparation
    """
    input_dir = config["data_prep"]["dir"]
    channel = config["data_prep"]["channel"]
    test_f = config["data_prep"]["test_fraction"]
    seed_split = config["data_prep"]["seed_split"]
    out_dir = config["output"]["directory"]
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    list_prompt, list_nonprompt, list_bkg = get_list_input_files(input_dir, channel)

    hdl_prompt = TreeHandler(list_prompt)
    hdl_nonprompt = TreeHandler(list_nonprompt)
    hdl_bkg = TreeHandler(list_bkg)

    df_prompt = hdl_prompt.get_data_frame()
    df_nonprompt = hdl_nonprompt.get_data_frame()
    df_bkg = hdl_bkg.get_data_frame()
    df_prompt.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_nonprompt.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_bkg.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prompt.dropna(inplace=True)
    df_nonprompt.dropna(inplace=True)
    df_bkg.dropna(inplace=True)

    n_prompt = len(df_prompt)
    n_nonprompt = len(df_nonprompt)
    n_bkg = len(df_bkg)
    print("\nNumber of available candidates: \n     "
          f"Prompt: {n_prompt}\n     FD: {n_nonprompt}\n     Bkg: {n_bkg}\n")

    df_tot = pd.concat([df_bkg, df_prompt, df_nonprompt], sort=True)
    labels_array = np.array([0]*n_bkg + [1]*n_prompt + [2]*n_nonprompt)
    if test_f < 1:
        train_set, test_set, y_train, y_test = train_test_split(
            df_tot, labels_array, test_size=test_f, random_state=seed_split
        )

    train_test_data = [train_set, y_train, test_set, y_test]
    del df_tot

    df_list = [df_bkg, df_prompt, df_nonprompt]
    leg_labels = ["bkg", "prompt", "nonprompt"]
    vars_dca = ["fDCAPrimXY1", "fDCAPrimZ1", "fDCAPrimXY2", "fDCAPrimZ2"]
    vars_pid = ["fNsigmaPiTPC1", "fNsigmaPiTPC2", "fNsigmaKaTPC1",
                "fNsigmaKaTPC2", "fNsigmaPiTOF1", "fNsigmaPiTOF2",
                "fNsigmaKaTOF1", "fNsigmaKaTPC2"]

    #_____________________________________________
    plot_utils.plot_distr(df_list, df_prompt.columns, 100, leg_labels,
                          figsize=(12, 7), alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f"{out_dir}/DistributionsAll_{channel}.pdf")
    plot_utils.plot_distr(df_list, vars_dca, 100, leg_labels,
                          figsize=(7, 7), alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f"{out_dir}/DistributionsImpPar_{channel}.pdf")
    plot_utils.plot_distr(df_list, vars_pid, 100, leg_labels,
                          figsize=(7, 7), alpha=0.3, log=True, grid=False, density=True)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(f"{out_dir}/DistributionsPID_{channel}.pdf")
    plt.close("all")

    #_____________________________________________
    corr_matrix_fig = plot_utils.plot_corr(df_list, df_prompt.columns, leg_labels)
    for fig, lab in zip(corr_matrix_fig, leg_labels):
        plt.figure(fig.number)
        plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9)
        fig.savefig(f"{out_dir}/CorrMatrix_{channel}_{lab}.pdf")

    return train_test_data


def train(config, train_test_data): #pylint: disable=too-many-locals
    """
    Function for the training

    Parameters
    -----------------
    - config: dictionary with configs
    - train_test_data: list with training and test data
    """

    out_dir = config["output"]["directory"]
    channel = config["data_prep"]["channel"]
    n_classes = len(np.unique(train_test_data[3]))
    model_clf = xgb.XGBClassifier(use_label_encoder=False)
    hyper_pars = config["ml"]["hyper_pars"]
    model_hdl = ModelHandler(model_clf, train_test_data[0].columns, hyper_pars)

    # train and test the model with the updated hyper-parameters
    y_pred_test = model_hdl.train_test_model(
        train_test_data,
        True,
        output_margin=config["ml"]["raw_output"],
        average=config["ml"]["roc_auc_average"],
        multi_class_opt=config["ml"]["roc_auc_approach"]
    )

    # save model handler in pickle
    model_hdl.dump_model_handler(f"{out_dir}/ModelHandler_{channel}.pickle")
    model_hdl.dump_original_model(f"{out_dir}/XGBoostModel_{channel}.model", True)

    #plots
    leg_labels = ["bkg", "prompt", "nonprompt"]

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 7)
    fig_ml_output = plot_utils.plot_output_train_test(
        model_hdl,
        train_test_data,
        80,
        config['ml']['raw_output'],
        leg_labels,
        True,
        density=True
    )

    if n_classes > 2:
        for fig, lab in zip(fig_ml_output, leg_labels):
            fig.savefig(f'{out_dir}/MLOutputDistr_{lab}_{channel}.pdf')
    else:
        fig_ml_output.savefig(f'{out_dir}/MLOutputDistr_{channel}.pdf')

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (10, 9)
    fig_ROC_curve = plot_utils.plot_roc(
        train_test_data[3],
        y_pred_test,
        None,
        leg_labels,
        config['ml']['roc_auc_average'],
        config['ml']['roc_auc_approach']
    )
    fig_ROC_curve.savefig(f'{out_dir}/ROCCurveAll_{channel}.pdf')
    pickle.dump(fig_ROC_curve, open(f'{out_dir}/ROCCurveAll_{channel}.pkl', 'wb'))

    #_____________________________________________
    plt.rcParams["figure.figsize"] = (12, 7)
    fig_feat_importance = plot_utils.plot_feature_imp(
        train_test_data[2][train_test_data[0].columns],
        train_test_data[3],
        model_hdl,
        leg_labels
    )
    n_plot = n_classes if n_classes > 2 else 1
    for i_fig, fig in enumerate(fig_feat_importance):
        if i_fig < n_plot:
            lab = leg_labels[i_fig] if n_classes > 2 else ''
            fig.savefig(f'{out_dir}/FeatureImportance_{lab}_{channel}.pdf')
        else:
            fig.savefig(f'{out_dir}/FeatureImportanceAll_{channel}.pdf')


def main(config):
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with configs
    """
    train_test_data = data_prep(config)
    train(config, train_test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", default="config_training.yml",
                        help="config file for training")
    args = parser.parse_args()

    with open(args.config, "r") as yml_cfg: #pylint: disable=bad-option-value
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    main(cfg)
