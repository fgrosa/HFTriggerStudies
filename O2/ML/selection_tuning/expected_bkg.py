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

def get_data_handler(file, branch, leaf_key):
    INFILE = os.path.join(os.getcwd(), file.file_path)

    return DataHandler(INFILE, var_name=leaf_key, treename=branch+leaf_key, limits=None, nbins=100)


#pylint: disable=redefined-outer-name
class ExpectedBackground:
    """
    Class to get expected background
    """
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.data = self.data_handler.get_binned_data_from_unbinned_data()
        self.bin_center = self.data_handler.get_bin_center()
        print("Init ExpectedBackground instance")

    @property
    def mean(self):
        """
        Helper task to compute the mean of the distribution
        """
        mean = 0
        for (value, center) in zip(self.data, self.bin_center):
            mean += value*center
        mean /= sum(self.data)
        return mean

    @property
    def sigma(self):
        """
        Helper task to compute the std deviation of the distribution
        """
        variance = 0
        for (value, center) in zip(self.data, self.bin_center):
            variance += value*(center - self.mean)**2
        variance /= (sum(self.data) - 1)
        sigma = np.sqrt(variance)
        return sigma

    @property
    def bin_width(self):
        """
        Helper task to compute the bin width
        """
        bin_width = []
        binning = self.data_handler.get_binned_obs_from_unbinned_data().binning[0]
        for bin_ in binning:
            bin_width.append(bin_[1] - bin_[0])
        return bin_width

    @property
    def area(self):
        """
        Helper task to compute area of histogram
        between mean-3sigma and mean+3sigma
        """
        area = 0
        integration_domain = (self.mean - 3*self.sigma, self.mean + 3*self.sigma)
        for (counts, center, width) in zip(self.data, self.bin_center, self.bin_width):
            if  integration_domain[0] <= center <= integration_domain[1]:
                area += counts*width
        return area

    @property
    def total_area(self):
        """
        Helper task to compute total area of histogram
        """
        area = 0
        for (counts, width) in zip(self.data, self.bin_width):
            area += counts*width
        return area

    @property
    def normalized_area(self):
        """
        Helper task to compute normalized area of histogram
        between mean-3sigma and mean+3sigma
        """
        return self.area/self.total_area

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
        return self.normalized_area*self.expected_events

"""
particleName = "Dplus"
file_name = "./AO2D.root"
file = uproot.open(file_name)

# keys to access invariant mass plots
branch, leaf_keys = get_keys(file, particleName)

print(leaf_keys)

# if there are several keys
# we need to create an instance for each key
expected_bkg = []
expected_bkg_instance = []
for key in leaf_keys:
    data_handler = get_data_handler(file, branch, key)
    expected_bkg.append(ExpectedBackground(data_handler).expected_background)
print(expected_bkg)
file.close()
"""
