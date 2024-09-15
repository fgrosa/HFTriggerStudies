"""
Compute expected background
"""

import os
import uproot
import numpy as np
import matplotlib
from flarefly.data_handler import DataHandler


def get_keys(file, particleName):
    branch_keys, leaf_keys = [], []
    for key in file.keys():
        if "train2p" in key:
            branch_keys.append(key)
        elif "train3p" in key:
            branch_keys.append(key)

    for branch_key in branch_keys:
        for leaf_key in file[branch_key].keys():
            if ("fInvMass" in leaf_key and particleName in leaf_key):
                leaf_keys.append(leaf_key)
                branch = branch_key + "/"
    return branch, leaf_keys

#pylint: disable=redefined-outer-name
class ExpectedBackground:
    """
    Class to get expected background
    """
    def __init__(self, df, key, eff, pt_bins, thresholds):
        self.df = df
        self.key = key
        self.eff = eff
        self.pt_bins = pt_bins
        self.thresholds = thresholds
        print("Init ExpectedBackground instance")

    @property
    def pt_prong(self):
        return "fPT2Prong" if "D0" in self.key else "fPT3Prong"

    @property
    def expected_events(self):
        """
        Helper task to compute the number of expected events
        """
        sigmapp = 60*1.e-3 #60 mb
        integrated_luminosity = 200*1.e+12 #200 pb^-1
        return sigmapp*integrated_luminosity

    @property
    def expected_background(self):
        """
        Helper task to compute the expected background
        """
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        exp_bkg = {}
        for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
            exp_bkg[(pt_min, pt_max)] = {}
            # select pt bin in dataframe
            df_pt = self.df.query(f"{pt_min} < {self.pt_prong} < {pt_max}")
            den = len(df_pt)
            # define the invariant mass window: mean+-3sigma
            mean, sigma = df_pt[self.key].mean(), df_pt[self.key].std()
            invmass_window = (mean - 3*sigma, mean + 3*sigma)
            # select invariant mass window
            df_invmass = df_pt.query(f"{invmass_window[0]} <= {self.key} <= {invmass_window[1]}")
            for iclas, clas in enumerate(["Bkg", "Prompt", "Nonprompt"]):
                exp_bkg[(pt_min, pt_max)][clas] = []
                for ithr, thr in enumerate(self.thresholds):
                    exp_bkg[(pt_min, pt_max)][clas].append(float(len(df_invmass))/den*self.expected_events)
                    # efficiency correction
                    exp_bkg[(pt_min, pt_max)][clas][ithr] *= self.eff[(pt_min, pt_max)][clas][ithr]

        return exp_bkg
