"""
Script for BDT efficiency calculation
"""

import argparse
import numpy as np
#import itertools
import pandas as pd
import yaml
import matplotlib.pyplot as plt

def config_bins(pt_bins):
    """
    Helper method to configure the bins to plot efficiency vs pT

    --------------------------------
    Parameters
    - pt_bins: array containing the pT values used for binning
    --------------------------------
    Outputs
    - bins_width: array with the width of each bin
    - bins_mean: array with the mean of each bin
    """
    pt_mins = pt_bins[:-1]
    pt_maxs = pt_bins[1:]
    bins_width = []
    bins_mean = []
    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        bins_width.append(0.5*(pt_max - pt_min))
        bins_mean.append(0.5*(pt_min + pt_max))
    return bins_width, bins_mean

def find_cut_indices(thresholds, BDT_cuts):
    """
    Helper method to find the index corresponding to chosen BDT cut in the 3D array eff

    --------------------------------
    Parameters
    - thresholds: array of BDT cuts to browse
    - BDT_cuts: array of chosen BDT output values for each class
    --------------------------------
    Outputs
    - effifiency: 3D array efficiency[pT index][class][efficiency value]
    """
    idx_list = []
    for cut in BDT_cuts:
        idx, = np.where(thresholds == cut)
        idx_list.append(idx[0])
    return idx_list

def compute_eff_vs_pt(eff, cuts_indices):
    """
    Helper method to compute efficiency vs pT
    
    --------------------------------
    Parameters
    - eff: output of compute_efficiency() method
    - cuts_indices: output of find_cuts_indices() method
    --------------------------------
    Outputs
    - effifiency: 3D array efficiency[pT index][class][efficiency value]
    """
    eff_vs_pt = []
    for clas in range(3):
        eff_vs_pt.append([])
        for ipt in range(len(eff)):
            eff_vs_pt[clas].append(eff[ipt][clas][cuts_indices[clas]])
    return eff_vs_pt

def compute_efficiency(df, pt_prong, pt_bins, BDT_cuts, thresholds):
    """
    Helper method to compute efficiency
    
    --------------------------------
    Parameters
    - df: dataframe extracted from input file
    - pt_prong: fPT2Prong for 2-prong channel and fPT3Prong for 3-prong channel
    - pt_bins: array containing the pT values used for binning
    - BDT_cuts: array of chosen BDT output values for each class
    - thresholds: array of BDT cuts to browse
    --------------------------------
    Outputs
    - effifiency: 3D array efficiency[pT index][class][efficiency value]
    """
    
    pt_mins = pt_bins[:-1]
    pt_maxs = pt_bins[1:]

    efficiency = []
    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        efficiency.append([])
        df_pt = df.query(f"{pt_min} < {pt_prong} < {pt_max}")
        den = len(df_pt)
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            operand = "<" if clas == "Bkg" else ">"
            efficiency[ipt].append([])
            for thr in thresholds:
                if iclas == 0:
                    df_cut = df_pt.query(f"ML_output_{clas} < {thr}")
                else:
                    df_cut = df_pt.query(f"ML_output_Bkg < {BDT_cuts[0]} and ML_output_{clas} > {thr}")
                efficiency[ipt][iclas].append(np.float(len(df_cut)) / den)

    return efficiency

def plot_efficiency_vs_BDTscore(eff, channel, pt_bins, BDT_cuts, thresholds):
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} BDT efficiencies", figsize=(16, 5))
    # choice of pt value
    ipt = 0
    for label in range(3):
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            plt.subplot(1,3,iclas+1)
            plt.scatter(thresholds, eff[label][0][iclas])
            plt.ylim(0.00001,1.)
            plt.yscale('log')
            plt.xlabel("BDT score")
            if iclas == 0:
                plt.ylabel(f"Efficiency for {clas} at pT = {pt_bins[ipt]} GeV/c")
            else:
                plt.ylabel(f"Efficiency for {clas} at pT = {pt_bins[ipt]} GeV/c (Bkg BDT < {BDT_cuts[0]})")
        plt.legend(['Bkg', 'Prompt', 'Nonprompt'])

    plt.show()

def plot_efficiency_vs_pt(eff_vs_pt, channel, pt_bins, BDT_cuts):
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} BDT efficiencies", figsize=(16, 5))
    # configure binning for the plot
    bins_width, bins_mean = config_bins(pt_bins)

    for label in range(3):
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            plt.subplot(1,3,iclas+1)
            plt.errorbar(bins_mean, eff_vs_pt[label][iclas], xerr = bins_width, fmt = 'o', markersize=5, elinewidth = 2, capsize=4)
            plt.ylim(0.001,1.)
            plt.yscale('log')
            plt.xlabel(r"$p_T$ (GeV/c)")
            if iclas == 0:
                plt.ylabel(f"Efficiency for {clas} BDT cut at {BDT_cuts[iclas]}")
            else:
                plt.ylabel(f"Efficiency for {clas} BDT cut at {BDT_cuts[iclas]} (Bkg BDT < {BDT_cuts[0]})")
        plt.legend(['Bkg', 'Prompt', 'Nonprompt'])

    plt.show()

def main(config):
    """
    Main function
    """

    infile = config["input_file"]
    channel = config["channel"]
    pt_bins = config["pt_bins"]
    bkg_BDT_cut = config["BDT_cut"]["Bkg"]
    prompt_BDT_cut = config["BDT_cut"]["Prompt"]
    nonprompt_BDT_cut = config["BDT_cut"]["Nonprompt"]
    n_points = config["n_points"]

    # store the indices corresponding to the BDT cuts for each class
    thresholds = np.linspace(0., 1., n_points)
    BDT_cuts = [bkg_BDT_cut, prompt_BDT_cut, nonprompt_BDT_cut]
    cuts_indices = find_cut_indices(thresholds, BDT_cuts)

    # distinguish between 2-prong and 3-prong channels
    pt_prong = "fPT2Prong" if channel == "D0ToKPi" else "fPT3Prong"

    # import file
    df = pd.read_parquet(infile)

    # compute efficiencies
    eff = []
    eff_vs_pt = []
    for label in range(3):
        eff.append([])
        eff_vs_pt.append([])
        eff[label] = compute_efficiency(df.query(f"Labels == {label}"), pt_prong, pt_bins, BDT_cuts, thresholds)
        eff_vs_pt[label] = compute_eff_vs_pt(eff[label], cuts_indices)
    
    plot_efficiency_vs_BDTscore(eff, channel, pt_bins, BDT_cuts, thresholds)
    plot_efficiency_vs_pt(eff_vs_pt, channel, pt_bins, BDT_cuts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config_efficiency.yml",
                        help="config file for efficiency plot")
    args = parser.parse_args()

    with open(args.config, "r") as yml_cfg:  # pylint: disable=bad-option-value
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    main(cfg)
