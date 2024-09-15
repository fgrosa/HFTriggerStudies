import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import uproot

import preselection # module for Preselection study
import ml_output # module for ML output study
import fonll_output # module for FONLL output study
import efficiency # module for total efficiencies computation
import yield_fraction # module to get fprompt and fnonprompt
from expected_signal import ExpectedSignal
import expected_bkg
import expected_signal_over_bkg

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
    prompt_total_efficiencies = efficiency.TotalEfficiencies(eval('{}_preselection_efficiencies_dico'.format('prompt')), BDT_efficiencies["Prompt"], pt_bins, thresholds).total_efficiencies
    nonprompt_total_efficiencies = efficiency.TotalEfficiencies(eval('{}_preselection_efficiencies_dico'.format('nonprompt')), BDT_efficiencies["Nonprompt"], pt_bins, thresholds).total_efficiencies

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
    fprompt, fnonprompt = yield_fraction.YieldFraction(prompt_fonll, nonprompt_fonll, prompt_total_efficiencies, nonprompt_total_efficiencies, pt_bins, thresholds).fprompt_fnonprompt
    """"
    EXPECTED SIGNAL
    """
    expectedSignal_instance = ExpectedSignal(prompt_fonll, fprompt, fnonprompt, prompt_total_efficiencies, pt_bins, thresholds)
    expected_signal = expectedSignal_instance.expected_signal

    if config["plot_options"]["expected_signal"]["plot"]:
        save = config["plot_options"]["expected_signal"]["save"]
        pt_bin = (config["pt_bin"][0], config["pt_bin"][1])
        expectedSignal_instance.plot_expected_signal_and_fractions(channel, prompt_fonll.number_of_events_list, pt_bin, save)

    """
    EXPECTED BACKGROUND
    """
    file_name = config["AO2D_file"]["file"]
    particleName = config["AO2D_file"]["particle_name"]
    AO2Dfile = uproot.open(file_name)
    # keys to access invariant mass plots
    branch, leaf_keys = expected_bkg.get_keys(AO2Dfile, particleName)
    # if there are several keys (i.e. several decay channels)
    # we need to create an instance for each key
    expected_bkg_list = []
    for leaf_key in leaf_keys:
        df = AO2Dfile[branch].arrays(library='pd')
        expected_bkg_list.append(expected_bkg.ExpectedBackground(df, leaf_key, BDT_efficiencies['Bkg'], pt_bins, thresholds).expected_background)
    # total expected background
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    total_expected_bkg = {}
    for pt_bin in zip(pt_mins, pt_maxs):
        total_expected_bkg[pt_bin] = {}
        for clas in ["Bkg", "Prompt", "Nonprompt"]:
            total_expected_bkg[pt_bin][clas] = [0]*len(thresholds)
            for ithr, _ in enumerate(thresholds):
                for exp_bkg in expected_bkg_list:
                    total_expected_bkg[pt_bin][clas][ithr] += exp_bkg[pt_bin][clas][ithr]

    """
    EXPECTED SIGNAL/BACKGROUND
    """
    s_over_b = expected_signal_over_bkg.ExpectedSignalOverBkg(expected_signal, total_expected_bkg, pt_bins, thresholds).s_over_b

    if config["plot_options"]["expected_signal_over_bkg"]["plot"]:
        save = config["plot_options"]["expected_signal_over_bkg"]["save"]
        pt_bin = (config["pt_bin"][0], config["pt_bin"][1])
        expected_signal_over_bkg.plot_s_over_b(s_over_b, channel, thresholds, pt_bin, save)
    

    # show figures
    plt.show()

    # close files
    preselectionFile.close()
    FONLL_file.close()
    AO2Dfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config_efficiency.yml",
                        help="config file for efficiency plot")
    args = parser.parse_args()

    with open(args.config, "r") as yml_cfg:  # pylint: disable=bad-option-value
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    main(cfg)