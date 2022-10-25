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
    def __init__(self, df):
        self.df = df
        print("Init ExpectedBackground instance")

    @property
    def selected_counts(self):
        """
        Helper task to compute the entries in the histogram
        between mean-3sigma and mean+3sigma
        """
        mean, sigma = self.df.mean()[0], self.df.std()[0]
        integration_domain = (mean - 3*sigma, mean + 3*sigma)
        name = self.df.axes[1][0]
        selected_df = self.df.query(f"{integration_domain[0]} <= {name} <= {integration_domain[1]}")
        return len(selected_df)

    @property
    def total_counts(self):
        """
        Helper task to compute the total number of entries in the histogram
        """
        return len(self.df)

    @property
    def normalized_area(self):
        """
        Helper task to compute normalized area of histogram
        between mean-3sigma and mean+3sigma
        """
        return self.selected_counts/self.total_counts

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
        return self.normalized_area*self.expected_events

"""
particleName = "D0"
file_name = "./AO2D.root"
file = uproot.open(file_name)
# keys to access invariant mass plots
branch, leaf_keys = get_keys(file, particleName)

expected_bkg_list = []
for leaf_key in leaf_keys:
    key = branch + '/' + leaf_key
    df = file[key].arrays(library='pd')
    expected_bkg_list.append(ExpectedBackground(df).expected_background)
# total expected background
exp_bkg = sum(exp_bkg_list)
"""
