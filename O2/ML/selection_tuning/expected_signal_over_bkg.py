import numpy as np
import matplotlib.pyplot as plt

class ExpectedSignalOverBkg:
    """
    Class to get expected signal/bkg
    """
    def __init__(self, exp_signal, exp_bkg, pt_bins, thresholds):
        self.exp_signal = exp_signal
        self.exp_bkg = exp_bkg
        self.pt_bins = pt_bins
        self.thresholds = thresholds
        print("Init ExpectedSignalOverBkg instance")

    @property
    def s_over_b(self):
        s_over_b = {}
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        for _, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
            s_over_b[pt_bin] = {}
            for _, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
                s_over_b[pt_bin][clas] = []
                for ithr, _ in enumerate(self.thresholds):
                    num = self.exp_signal[pt_bin][clas][ithr]
                    denom = self.exp_bkg[pt_bin][clas][ithr]
                    s_over_b[pt_bin][clas].append(num/denom) if denom!=0 else s_over_b[pt_bin][clas].append(0)

        return s_over_b

def plot_s_over_b(s_over_b, channel, thresholds, pt_bin, save):
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} S/B", figsize=(16, 5))
    # choice of pt interval
    pt_min = pt_bin[0]
    pt_max = pt_bin[1]
    # list filled with non-zero min and max of y axis values 
    # to select a common y axis window for all the subplots
    y_range = []
    for clas in ["Bkg", "Prompt", "Nonprompt"]:
        y_range.append(min(i for i in s_over_b[pt_bin][clas] if i!=0))
        y_range.append(max(s_over_b[pt_bin][clas]))
    # plotting S/B
    for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
        plt.subplot(1, 3, iclas+1)
        plt.xlabel(f"ML output {clas} BDT score")
        plt.ylabel(f"Expected signal/background for {pt_min} < pT < {pt_max} GeV/c") if iclas == 0 else None
        plt.xlim(0, 1)
        plt.ylim(min(y_range)/10, max(y_range)*10)
        plt.yscale('log')
        plt.scatter(thresholds, s_over_b[pt_bin][clas])
        
    if save:
        plt.savefig(f"./{channel}_expected_s_over_bkg_for_pt_in_{pt_min}_{pt_max}.png")
        plt.close("all")
