import matplotlib.pyplot as plt

class ExpectedSignal:
    """
    Class to get expected signal
    """
    def __init__(self, prompt_fonll, fprompt, fnonprompt, prompt_total_eff, pt_bins, thresholds):
        self.prompt_fonll = prompt_fonll
        self.fprompt = fprompt
        self.fnonprompt = fnonprompt
        self.prompt_total_eff = prompt_total_eff
        self.pt_bins = pt_bins
        self.thresholds = thresholds
        print("Init ExpectedSignal instance")

    @property
    def expected_signal(self):
        """
        Helper method to compute expected signal

        --------------------------------
        Parameters
        - prompt_fonll: FONLL_output object (needed to set pt_bin value)
        - fprompt: prompt fraction (3D aray)
        - prompt_total_eff: total efficiency (preselection*BDT) for prompt (3D array)
        - pt_bins: chosen binning for pT
        - thresholds: range of BDT scores
        --------------------------------
        Outputs
        - expected_signal: expected_signal[pt_bin][clas][ithr] (3D array)
        """
        expected_signal = {}
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
            expected_signal[pt_bin] = {}
            self.prompt_fonll.pt_bin = pt_bin
            pt_min, pt_max = pt_bin[0], pt_bin[1]
            Delta_pt = pt_max - pt_min
            for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
                expected_signal[pt_bin][clas] = []
                for ithr, _ in enumerate(self.thresholds):
                    expected_signal[pt_bin][clas].append([])
                    if self.fprompt[pt_bin][clas][ithr] != 0:
                        expected_signal[pt_bin][clas][ithr] = 2*self.prompt_fonll.integrated_dsigma_dpt_BR*self.prompt_total_eff[pt_bin][clas][ithr]*Delta_pt*self.prompt_fonll.luminosity/self.fprompt[pt_bin][clas][ithr]
                    else:
                        expected_signal[pt_bin][clas][ithr] = 0
        return expected_signal

    def plot_expected_signal_and_fractions(self, channel, Nev_list, pt_bin, save):
        plt.rcParams['axes.grid'] = True
        # choice of pt interval
        pt_min, pt_max = pt_bin[0], pt_bin[1]
        plt.figure(f"{channel} for {pt_min} < pT < {pt_max} (GeV/c)", figsize=(16, 10))
        # list filled with non-zero min of y axis values 
        # to select a common y axis window for all the subplots
        y_range = []
        for clas in ["Bkg", "Prompt", "Nonprompt"]:
            y_range.append(min(i for i in self.expected_signal[pt_bin][clas] if i!=0))
        for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']*2):
            plt.subplot(2, 3, iclas+1)
            plt.xlabel(f"ML output {clas} BDT score")
            plt.xlim(0, 1)
            if iclas <= 2:
                plt.scatter(self.thresholds, self.expected_signal[pt_bin][clas], label="Expected signal")
                plt.plot(self.thresholds, Nev_list, color='r', label=r"$N_{ev}$")
                #plt.ylabel("Expected signal")
                plt.ylim(min(y_range)/10, 5*Nev_list[0])
                plt.yscale('log')
                if iclas == 2: plt.legend(loc="best")
            else:
                plt.scatter(self.thresholds, self.fprompt[pt_bin][clas], label="fprompt")
                plt.scatter(self.thresholds, self.fnonprompt[pt_bin][clas], label="fnonprompt")
                if iclas == 5: plt.legend(loc="best")
        if save:
            plt.savefig(f"./{channel}_expected_signal_for_pt_in_{pt_min}_{pt_max}.png")
            plt.close("all")