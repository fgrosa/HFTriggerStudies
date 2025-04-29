import numpy as np
import matplotlib.pyplot as plt

class TotalEfficiencies:
    """
    Class to get BDT efficiency times preselection efficiency
    """
    def __init__(self, preselection_efficiencies, BDT_efficiencies, pt_bins, thresholds):
        self.preselection_efficiencies = preselection_efficiencies
        self.BDT_efficiencies = BDT_efficiencies
        self.pt_bins = pt_bins
        self.thresholds = thresholds
        print("Init TotalEfficiencies instance")

    @property
    def total_efficiencies(self):
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
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        for pt_bin in zip(pt_mins, pt_maxs):
            total_efficiencies[pt_bin] = {}
            for clas in ['Bkg', 'Prompt', 'Nonprompt']:
                total_efficiencies[pt_bin][clas] = []
                for ithr, _ in enumerate(self.thresholds):
                    total_efficiencies[pt_bin][clas].append([])
                    total_efficiencies[pt_bin][clas][ithr] = self.preselection_efficiencies[pt_bin] * self.BDT_efficiencies[pt_bin][clas][ithr]
        return total_efficiencies
