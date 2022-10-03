import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

def integrate_dsigma_dpt_BR(self):
    """
    Method to sum all dsigma/dpt*BR bins over a given pT interval
    """
    dsigma_dpt_BR = self.data_array[0]
    pt = self.data_array[1]
    pt_min, pt_max = self.pt_bin[0], self.pt_bin[1]
    # list of indices corresponding to Delta_pt interval
    idx_list, = np.where( (pt >= pt_min) & (pt <= pt_max) )
    # select data in this interval
    boolean = []
    for i, _ in enumerate(dsigma_dpt_BR):
        boolean.append(True) if i in idx_list else boolean.append(False)
    # integrate dsigma_dpt_BR over Delta_pt interval
    sum = np.sum(dsigma_dpt_BR, where = boolean)
    return sum

def get_channel_key(self):
    """
    Method to get branch name for our channel in FONLL file
    """
    keys = self.FONLL_file.keys()
    for key in keys:
        if self.channel+"pred" in key and "max" in key:
            prompt_key = key
        if self.channel in key and "Bpred" in key and "central" in key:
            nonprompt_key = key
    if (self.promptness == "prompt"): return prompt_key
    else: return nonprompt_key



class FONLL_output:
    """
    Class to extract necessary information (dsigma/dpt) useful to compute the expected signal
    """

    def __init__(self, promptness, FONLL_file, channel, pt_bins, n_points):
        self.promptness = promptness
        self.FONLL_file = FONLL_file
        self.channel = channel
        self.pt_bins = pt_bins
        self.n_points = n_points

    @property
    def thresholds(self):
        return np.linspace(0., 1., self.n_points)
    @property
    def luminosity(self):
        return 200 # pb^-1
    @property
    def cross_section(self):
        mub_to_pb = 1.e-6 / 1e-12
        sigma = 60 # mub
        return sigma * mub_to_pb # pb
    @property
    def number_of_events(self):
        return self.luminosity * self.cross_section
    @property
    def number_of_events_list(self):
        return [self.number_of_events]*len(self.thresholds)

    @property
    def channel_key(self):
        return get_channel_key(self)
    @property
    def data_array(self):
        return self.FONLL_file[self.channel_key].to_hist().to_numpy()
    @property
    def dsigma_dpt_BR(self):
        return self.data_array[0]
    @property
    def integrated_dsigma_dpt_BR(self):
        return integrate_dsigma_dpt_BR(self)

    @property
    def pt_bin(self):
        return self._pt_bin
    @pt_bin.setter
    def pt_bin(self, pt_bin):
        self._pt_bin = pt_bin