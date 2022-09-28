import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot


def config_bins(pt_bins):
    """
    Helper method to configure the bins to plot efficiency vs pT

    --------------------------------
    Parameter
    - pt_bins: array containing the pT values used for binning
    --------------------------------
    Outputs
    - bins_width: array with the width of each bin
    - bins_mean: array with the mean of each bin
    """
    pt_mins, pt_maxs = pt_bins[:-1], pt_bins[1:]
    bins_width = []
    bins_mean = []
    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        bins_width.append(0.5*(pt_max - pt_min))
        bins_mean.append(0.5*(pt_min + pt_max))
    return bins_width, bins_mean

def find_cut_indices(self):
    """
    Helper method to find the index corresponding to chosen BDT cut in the 3D array eff

    --------------------------------
    Output
    - idx_list: index corresponding to chosen BDT cut in the 3D array eff
    """
    idx_list = {}
    for clas in self.BDT_cuts:
        idx, = np.where(self.thresholds == self.BDT_cuts[clas])
        idx_list[clas] = idx[0]
    return idx_list

def compute_efficiencies(self):
    """
    Helper method to compute efficiencies

    --------------------------------
    Output
    - effifiencies: 3D array efficiencies[pT bin][class][ithr]
    """
    pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
    efficiencies = {}
    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        efficiencies[(pt_min, pt_max)] = {}
        df_pt = self.df.query(f"{pt_min} < {self.pt_prong} < {pt_max}")
        den = len(df_pt)
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            operand = "<" if clas == "Bkg" else ">"
            efficiencies[(pt_min, pt_max)][clas] = []
            for thr in self.thresholds:
                if clas == "Bkg":
                    df_cut = df_pt.query(f"ML_output_{clas} < {thr}")
                else:
                    df_cut = df_pt.query(f"ML_output_Bkg < {self.BDT_cuts['Bkg']} and ML_output_{clas} > {thr}")
                efficiencies[(pt_min, pt_max)][clas].append(float(len(df_cut)) / den)
    return efficiencies

def _compute_efficiencies_at_BDT_cut(self, eff):
    """
    Helper method to compute efficiencies vs pT
    
    --------------------------------
    Parameter
    - eff: output of compute_efficiencies() method
    --------------------------------
    Output
    - eff_at_BDT_cut: 3D array efficiencies[pT index][class][ithr]
    """
    pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
    eff_at_BDT_cut = {}
    for iclas, clas in enumerate(self.cuts_indices):
        eff_at_BDT_cut[clas] = []
        for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
            eff_at_BDT_cut[clas].append(eff[(pt_min, pt_max)][clas][self.cuts_indices[clas]])
    return eff_at_BDT_cut



class ML_output:
    """
    Class ...
    true ilabel, ML_output file, pt_bins, channel, BDT_cuts, n_points, thresholds
    """
    def __init__(self, ilabel, ML_output_file, channel, pt_bins, BDT_cuts, n_points):
        self.ilabel = ilabel
        self.ML_output_file = ML_output_file
        self.channel = channel
        self.pt_bins = pt_bins
        self.BDT_cuts = BDT_cuts
        self.n_points = n_points

    @property
    def thresholds(self):
        return np.linspace(0., 1., self.n_points)
    @property
    def cuts_indices(self):
        return find_cut_indices(self)
    @property
    def pt_prong(self):
        return "fPT2Prong" if self.channel == "D0ToKPi" else "fPT3Prong"
    @property
    def df(self):
        return pd.read_parquet(self.ML_output_file).query(f"Labels == {self.ilabel}")
    
    @property
    def efficiencies(self):
        return compute_efficiencies(self)

    # too slow because asks for recomputing of efficiencies
    """
    @property
    def efficiencies_at_BDT_cut(self):
        return compute_efficiencies_at_BDT_cut(self)
    """
    # faster because one uses the efficiencies computed a priori
    compute_efficiencies_at_BDT_cut = _compute_efficiencies_at_BDT_cut
    



def plot_efficiency_vs_BDTscore(eff, channel, pt_bins, BDT_cuts, thresholds, pt_bin, save):
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} BDT efficiencies", figsize=(16, 5))
    # choice of pt interval
    pt_min = pt_bin[0]
    pt_max = pt_bin[1]
    for ilabel, label in enumerate(["Bkg", "Prompt", "Nonprompt"]):
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            plt.subplot(1, 3, iclas+1)
            plt.scatter(thresholds, eff[label][pt_bin][clas])
            plt.ylim(0.00001,1.)
            plt.yscale('log')
            plt.xlabel("BDT score")
            if iclas == 0:
                plt.ylabel(f"Efficiency for {clas} for {pt_min} < pT < {pt_max} GeV/c")
            else:
                plt.ylabel(f"Efficiency for {clas} for {pt_min} < pT < {pt_max} GeV/c (Bkg BDT < {BDT_cuts['Bkg']})")
        plt.legend(['Bkg', 'Prompt', 'Nonprompt'], loc="best")
    if save:
        plt.savefig(f"./{channel}_efficiency_vs_BDTscore_for_pt_in_{pt_min}_{pt_max}.png")
        plt.close("all")

def plot_efficiency_vs_pt(eff_at_BDT_cut, channel, pt_bins, BDT_cuts, save):
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} BDT efficiencies vs pT", figsize=(16, 5))
    # configure binning for the plot
    bins_width, bins_mean = config_bins(pt_bins)

    for ilabel, label in enumerate(["Bkg", "Prompt", "Nonprompt"]):
        for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
            plt.subplot(1, 3, iclas+1)
            plt.errorbar(bins_mean, eff_at_BDT_cut[label][clas], xerr = bins_width, fmt = 'o', markersize=5, elinewidth = 2, capsize=4)
            plt.ylim(0.001,1.)
            plt.yscale('log')
            plt.xlabel(r"$p_\mathrm{T}$ (GeV/$c$)")
            if iclas == 0:
                plt.ylabel(f"Efficiency for {clas} BDT cut at {BDT_cuts[clas]}")
            else:
                plt.ylabel(f"Efficiency for {clas} BDT cut at {BDT_cuts[clas]} (Bkg BDT < {BDT_cuts['Bkg']})")
        plt.legend(['Bkg', 'Prompt', 'Nonprompt'], loc="best")
    if save:
        plt.savefig(f"./{channel}_BDT_efficiency_vs_pt.png")
        plt.close("all")