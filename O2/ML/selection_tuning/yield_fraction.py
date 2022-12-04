import numpy as np
import matplotlib.pyplot as plt

class YieldFraction:
    """
    Class to get (non)prompt fraction of yield
    """
    def __init__(self, prompt_fonll, nonprompt_fonll, prompt_total_eff, nonprompt_total_eff, pt_bins, thresholds):
        self.prompt_fonll = prompt_fonll
        self.nonprompt_fonll = nonprompt_fonll
        self.prompt_total_eff = prompt_total_eff
        self.nonprompt_total_eff = nonprompt_total_eff
        self.pt_bins = pt_bins
        self.thresholds = thresholds
        print("Init YieldFraction instance")

    @property
    def fprompt_fnonprompt(self):
        """
        Helper method to compute prompt and nonprompt fractions

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
        - tuple fprompt, fnonprompt: (non)prompt fraction f(non)prompt[pt_bin][clas][ithr] (3D array)
        """
        fprompt, fnonprompt = {}, {}
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
            fprompt[pt_bin], fnonprompt[pt_bin] = {}, {}
            self.prompt_fonll.pt_bin = pt_bin
            self.nonprompt_fonll.pt_bin = pt_bin
            for iclas, clas in enumerate(['Bkg', 'Prompt', 'Nonprompt']):
                fprompt[pt_bin][clas], fnonprompt[pt_bin][clas] = [], []
                for ithr, _ in enumerate(self.thresholds):
                    fprompt[pt_bin][clas].append([])
                    fnonprompt[pt_bin][clas].append([])
                    num = self.nonprompt_total_eff[pt_bin][clas][ithr] * self.nonprompt_fonll.integrated_dsigma_dpt_BR
                    denom = self.prompt_total_eff[pt_bin][clas][ithr] * self.prompt_fonll.integrated_dsigma_dpt_BR
                    if denom !=0:
                        fprompt[pt_bin][clas][ithr] = 1 / (1 + num / denom)
                    else:
                        fprompt[pt_bin][clas][ithr] = 0
                    fnonprompt[pt_bin][clas][ithr] = 1 - fprompt[pt_bin][clas][ithr] 
        return fprompt, fnonprompt
              