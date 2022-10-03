import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import uproot

import preselection # module for Preselection study
import ml_output # module for ML output study
import fonll_output # module for FONLL output study


def compute_total_efficiencies(preselection_eff, BDT_eff, pt_bins, thresholds):
    """
    Helper method to compute total efficiencies

    --------------------------------
    Parameters
    - preselection_eff: preselection efficiencies (1D array)
    - BDT_eff: BDT efficiencies (3D array)
    - pt_bins: chosen binning for pT
    - thresholds: range of BDT scores
    --------------------------------
    Outputs
    - total_efficiencies: 3D array total_efficiencies[pt_bin][clas][ithr]
    """
    total_efficiencies = {}
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
        total_efficiencies[pt_bin] = {}
        for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
            total_efficiencies[pt_bin][clas] = []
            for ithr, _ in enumerate(thresholds):
                total_efficiencies[pt_bin][clas].append([])
                total_efficiencies[pt_bin][clas][ithr] = preselection_eff[pt_bin] * BDT_eff[pt_bin][clas][ithr]
    return total_efficiencies


def compute_fprompt(prompt_fonll, nonprompt_fonll, prompt_total_eff, nonprompt_total_eff, pt_bins, thresholds):
    """
    Helper method to compute prompt fraction

    --------------------------------
    Parameters
    - prompt_fonll: FONLL_output object (needed to set pt_bin value)
    - nonprompt_fonll: FONLL_output object (needed to set pt_bin value)
    - prompt_total_eff: total efficiency (preselection*BDT) for prompt (3D array)
    - nonprompt_total_eff: total efficiency (preselection*BDT) for nonprompt (3D array)
    - pt_bins: chosen binning for pT
    - thresholds: range of BDT scores
    --------------------------------
    Outputs
    - fprompt: prompt fraction fprompt[pt_bin][clas][ithr] (3D array)
    """
    fprompt = {}
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
        fprompt[pt_bin] = {}
        prompt_fonll.pt_bin = pt_bin
        nonprompt_fonll.pt_bin = pt_bin
        for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
            fprompt[pt_bin][clas] = []
            for ithr, _ in enumerate(thresholds):
                fprompt[pt_bin][clas].append([])
                num = nonprompt_total_eff[pt_bin][clas][ithr] * nonprompt_fonll.integrated_dsigma_dpt_BR
                denom = prompt_total_eff[pt_bin][clas][ithr] * prompt_fonll.integrated_dsigma_dpt_BR
                if denom !=0:
                    fprompt[pt_bin][clas][ithr] = 1 / (1 + num / denom)
                else:
                    fprompt[pt_bin][clas][ithr] = 0
    return fprompt
                    
def compute_fnonprompt(fprompt, pt_bins, thresholds):
    """
    Helper method to basically perform "fnonprompt = 1 - fprompt"
    --------------------------------
    Outputs
    - fnonprompt: nonprompt fraction fnonprompt[pt_bin][clas][ithr] (3D array)
    """
    fnonprompt = {}
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
        fnonprompt[pt_bin] = {}
        for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
            fnonprompt[pt_bin][clas] = []
            for ithr, _ in enumerate(thresholds):
                fnonprompt[pt_bin][clas].append([])
                fnonprompt[pt_bin][clas][ithr] = 1 - fprompt[pt_bin][clas][ithr]
    return fnonprompt                


def compute_expected_signal(prompt_fonll, fprompt, fnonprompt, prompt_total_eff, pt_bins, thresholds):
    """
    Helper method to compute expected signal

    --------------------------------
    Parameters
    - prompt_fonll: FONLL_output object (needed to set pt_bin value)
    - fprompt: prompt fraction (3D aray)
    - fnonprompt: nonprompt fraction (3D array)
    - prompt_total_eff: total efficiency (preselection*BDT) for prompt (3D array)
    - pt_bins: chosen binning for pT
    - thresholds: range of BDT scores
    --------------------------------
    Outputs
    - bins_width: array with the width of each bin
    - bins_mean: array with the mean of each bin
    """
    expected_signal = {}
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
        expected_signal[pt_bin] = {}
        prompt_fonll.pt_bin = pt_bin
        pt_min, pt_max = pt_bin[0], pt_bin[1]
        Delta_pt = pt_max - pt_min
        for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
            expected_signal[pt_bin][clas] = []
            for ithr, _ in enumerate(thresholds):
                expected_signal[pt_bin][clas].append([])
                if fprompt[pt_bin][clas][ithr] != 0:
                    expected_signal[pt_bin][clas][ithr] = 2 * prompt_fonll.integrated_dsigma_dpt_BR * prompt_total_eff[pt_bin][clas][ithr] * Delta_pt * prompt_fonll.luminosity / fprompt[pt_bin][clas][ithr]
                else:
                    expected_signal[pt_bin][clas][ithr] = 0
    return expected_signal


def plot_expected_signal_and_fractions(channel, expected_signal, fprompt, fnonprompt, Nev_list, pt_bins, thresholds, pt_bin, save):
    plt.rcParams['axes.grid'] = True
    # choice of pt interval
    pt_min, pt_max = pt_bin[0], pt_bin[1]
    plt.figure(f"{channel} for {pt_min} < pT < {pt_max} (GeV/c)", figsize=(16, 10))
    for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']*2):
        plt.subplot(2, 3, iclas+1)
        plt.xlabel(f"ML output {clas} BDT score")
        plt.xlim(0, 1)
        if iclas <= 2:
            plt.scatter(thresholds, expected_signal[pt_bin][clas], label="Expected signal")
            plt.plot(thresholds, Nev_list, color='r', label=r"$N_{ev}$")
            #plt.ylabel("Expected signal")
            plt.ylim(1, 5*Nev_list[0])
            plt.yscale('log')
            if iclas == 2: plt.legend(loc="best")
        else:
            plt.scatter(thresholds, fprompt[pt_bin][clas], label="fprompt")
            plt.scatter(thresholds, fnonprompt[pt_bin][clas], label="fnonprompt")
            if iclas == 5: plt.legend(loc="best")
    if save:
        plt.savefig(f"./{channel}_expected_signal_for_pt_in_{pt_min}_{pt_max}.png")
        plt.close("all")


def main(config):
    """
    Main function
    """

    pt_bins = config["pt_bins"]

    """
    PRESELECTION
    """
    channel = config["Preselection"]["channel"]
    # open file
    preselectionFile = uproot.open(config["Preselection"]["file"]["name"])
    # extract TTrees
    tree_mc_rec = preselectionFile[config["Preselection"]["file"]["tree_mc_rec"]]
    tree_mc_gen = preselectionFile[config["Preselection"]["file"]["tree_mc_gen"]]
    # Preselection objects
    for promptness in ["prompt", "nonprompt"]:
        exec(f"{promptness}_preselection = preselection.Preselection(channel, promptness, pt_bins, tree_mc_rec, tree_mc_gen)")
        exec(f"{promptness}_preselection_efficiencies_array = {promptness}_preselection.efficiencies_array")
        exec(f"{promptness}_preselection_efficiencies_dico = {promptness}_preselection.efficiencies_dico")
    if config["plot_options"]["preselection_efficiency"]["plot"]:
        save = config["plot_options"]["preselection_efficiency"]["save"]
        preselection.plot_preselection_efficiency(channel, eval('{}_preselection'.format('prompt')), eval('{}_preselection_efficiencies_array'.format('prompt')), eval('{}_preselection_efficiencies_array'.format('nonprompt')), pt_bins, save)

    """
    ML
    """
    # ML_output objects
    ML_output_file = config["ML_output"]["file"]
    channel = config["ML_output"]["channel"]
    BDT_cuts = config["ML_output"]["BDT_cuts"]
    n_points = config["ML_output"]["n_points"]
    thresholds = np.linspace(0., 1., n_points)
    # produce ML_output objects and compute BDT efficiencies
    BDT_efficiencies = {}
    BDT_efficiencies_at_BDT_cut = {}
    for ilabel, label in enumerate(["Bkg", "Prompt", "Nonprompt"]):
        exec(f"{label}_ML_output = ml_output.ML_output(ilabel, ML_output_file, channel, pt_bins, BDT_cuts, n_points)")
        BDT_efficiencies[label] = eval('{}_ML_output'.format(label)).efficiencies
        BDT_efficiencies_at_BDT_cut[label] = eval('{}_ML_output'.format(label)).compute_efficiencies_at_BDT_cut(BDT_efficiencies[label])
    # plot efficiencies
    if config["plot_options"]["BDT_efficiency_vs_BDTscore"]["plot"]:
        save = config["plot_options"]["BDT_efficiency_vs_BDTscore"]["save"]
        pt_bin = (config["pt_bin"][0], config["pt_bin"][1])
        ml_output.plot_efficiency_vs_BDTscore(BDT_efficiencies, channel, pt_bins, BDT_cuts, thresholds, pt_bin, save)
    if config["plot_options"]["BDT_efficiency_vs_pt"]["plot"]:
        save = config["plot_options"]["BDT_efficiency_vs_pt"]["save"]
        pt_bin = (config["pt_bin"][0], config["pt_bin"][1])
        ml_output.plot_efficiency_vs_pt(BDT_efficiencies_at_BDT_cut, channel, pt_bins, BDT_cuts, save)


    """ 
    TOTAL EFFICIENCIES
    """
    prompt_total_efficiencies = compute_total_efficiencies(eval('{}_preselection_efficiencies_dico'.format('prompt')), BDT_efficiencies["Prompt"], pt_bins, thresholds)
    nonprompt_total_efficiencies = compute_total_efficiencies(eval('{}_preselection_efficiencies_dico'.format('nonprompt')), BDT_efficiencies["Nonprompt"], pt_bins, thresholds)
    
    """
    FONLL
    """
    # open file with FONLL content
    FONLL_file = uproot.open(config["FONLL_output"]["file"])
    channel = config["FONLL_output"]["channel"]
    # FONLL_output objects
    prompt_fonll = fonll_output.FONLL_output("prompt", FONLL_file, channel, pt_bins, n_points)
    nonprompt_fonll = fonll_output.FONLL_output("nonprompt", FONLL_file, channel, pt_bins, n_points)

    """
    YIELD FRACTIONS
    """
    fprompt = compute_fprompt(prompt_fonll, nonprompt_fonll, prompt_total_efficiencies, nonprompt_total_efficiencies, pt_bins, thresholds)
    fnonprompt = compute_fnonprompt(fprompt, pt_bins, thresholds)
    """"
    EXPECTED SIGNAL
    """
    expected_signal = compute_expected_signal(prompt_fonll, fprompt, fnonprompt, prompt_total_efficiencies, pt_bins, thresholds)

    if config["plot_options"]["expected_signal"]["plot"]:
        save = config["plot_options"]["expected_signal"]["save"]
        pt_bin = (config["pt_bin"][0], config["pt_bin"][1])
        plot_expected_signal_and_fractions(channel, expected_signal, fprompt, fnonprompt, prompt_fonll.number_of_events_list, pt_bins, thresholds, pt_bin, save)
    
    # show figures
    plt.show()

    # close files
    preselectionFile.close()
    FONLL_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config_efficiency.yml",
                        help="config file for efficiency plot")
    args = parser.parse_args()

    with open(args.config, "r") as yml_cfg:  # pylint: disable=bad-option-value
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    main(cfg)