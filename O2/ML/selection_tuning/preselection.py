import uproot
import numpy as np
import matplotlib.pyplot as plt

from ml_output import config_bins

def get_rec_keys(self, keys):
    """
    Helper method to get the branch names for reconstructed particles in Preselection file
    """
    key_list = []
    for key in keys:
        if ("PtReco" in key and self.particleName in key):
            key_list.append(key)
    for key in key_list:
        if ("NonPrompt" in key):
            nonprompt_key = key
        else:
            prompt_key = key
    if (self.promptness == "prompt"): return prompt_key
    else: return nonprompt_key

def get_gen_keys(self, keys):
    """
    Helper method to get the branch names for generated particles in Preselection file
    """
    key_list = []
    for key in keys:
        if ("PtDistr" in key):
            key_list.append(key)
    for key in key_list:
        if ("NonPrompt" in key):
            nonprompt_key = key
        else:
            prompt_key = key
    if (self.promptness == "prompt"): return prompt_key
    else: return nonprompt_key

def get_pt_bins_correspondance(self, X_pt_bins):
    """
    Helper method to get pT binning correspondance between rec/gen binning and the binning in config file
    """
    idx_min_list, idx_max_list = [], []
    pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
    bin_width = X_pt_bins[1] - X_pt_bins[0]
    for ipt, (pt_min, pt_max) in enumerate(zip(pt_mins, pt_maxs)):
        idx_min, = np.where( ((X_pt_bins >= pt_min - 0.1*bin_width) & (X_pt_bins <= pt_min + 0.1*bin_width)))
        idx_max, =  np.where( ((X_pt_bins >= pt_max - 0.1*bin_width) & (X_pt_bins <= pt_max + 0.1*bin_width)))
        idx_min_list.append(idx_min[0])
        idx_max_list.append(idx_max[0])
    return idx_min_list, idx_max_list

def compute_yield_per_pt(self, array, idx_lists):
    """
    Helper method to get yield for a given pT bin
    """
    pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
    idx_min_list, idx_max_list = idx_lists[0], idx_lists[1]
    number_list = {}
    for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
        #a[i:j+1].sum() sum of all elements in a between indices i and j
        sum = array[ idx_min_list[ipt]:idx_max_list[ipt]+1 ].sum()
        number_list[pt_bin] = sum
    return number_list

class Preselection:
    """
    Preselection class to get the efficiencies
    """
    particles_dico = ["D0", "Dplus", "Ds", "Lc", "Xic"]
    channels_dico = ["D0ToKPi", "DplusToPiKPi", "DsToKKPi", "LcToPKPi", "XicToPKPi"]
    gen_particles_dico = ["D^{0}", "D^{#plus}", "D_{s}^{#plus}", "#Lambda_{c}^{#plus} #rightarrow pK^{#minus}#pi^{#plus}", "#Xi_{c}^{#plus} #rightarrow pK^{#minus}#pi^{#plus}"]

    def __init__(self, channel, promptness, pt_bins, tree_mc_rec, tree_mc_gen):
        self.channel = channel
        self.promptness = promptness
        self.pt_bins = pt_bins
        self.tree_mc_rec = tree_mc_rec
        self.tree_mc_gen = tree_mc_gen
        for name in self.particles_dico:
            if name in channel:
                self.particleName = name
        print('Init Preselection instance')
    @property
    def rec_key(self):
        return get_rec_keys(self, self.tree_mc_rec.keys())
    @property
    def gen_key(self):
        return get_gen_keys(self, self.tree_mc_gen.keys())
    @property
    def rec_array(self):
        return self.tree_mc_rec[self.rec_key].to_hist().to_numpy()[0]
    @property
    def rec_pt_bins(self):
        return self.tree_mc_rec[self.rec_key].to_hist().to_numpy()[1]
    
    @property
    def gen_particles_labels(self):
        gen_prompt_TH2D = self.tree_mc_gen[self.gen_key]
        gen_prompt_TH2D_x_labels = []
        for x_label in gen_prompt_TH2D.axis(0):
            gen_prompt_TH2D_x_labels.append(x_label)
        return gen_prompt_TH2D_x_labels
    @property
    def gen_particle_idx(self):
        channel_idx = self.channels_dico.index(self.channel)
        particle_idx = self.gen_particles_labels.index(self.gen_particles_dico[channel_idx])
        return particle_idx
    @property
    def gen_array(self):
        return self.tree_mc_gen[self.gen_key].to_hist().to_numpy()[0][self.gen_particle_idx]
    @property
    def gen_pt_bins(self):
        return self.tree_mc_gen[self.gen_key].to_hist().to_numpy()[2]

    @property
    def rec_idx_lists(self):
        return get_pt_bins_correspondance(self, self.rec_pt_bins)
    @property
    def gen_idx_lists(self):
        return get_pt_bins_correspondance(self, self.gen_pt_bins)

    @property
    def rec_yield(self):
        return compute_yield_per_pt(self, self.rec_array, self.rec_idx_lists)
    @property
    def gen_yield(self):
        return compute_yield_per_pt(self, self.gen_array, self.gen_idx_lists)

    # preselection efficiency = (rec MC) / (gen MC)
    @property
    def efficiencies_dico(self):
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        efficiencies = {}
        for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
            efficiencies[pt_bin] = self.rec_yield[pt_bin] / self.gen_yield[pt_bin]
        return efficiencies
    @property
    def efficiencies_array(self):
        pt_mins, pt_maxs = self.pt_bins[:-1], self.pt_bins[1:]
        efficiencies = []
        for ipt, pt_bin in enumerate(zip(pt_mins, pt_maxs)):
            efficiencies.append(self.rec_yield[pt_bin] / self.gen_yield[pt_bin])
        return efficiencies


def plot_preselection_efficiency(channel, preselection_object, prompt_preselection_efficiencies_array, nonprompt_preselection_efficiencies_array, pt_bins, save):
    bins_width, bins_mean = config_bins(pt_bins)
    plt.rcParams['axes.grid'] = True
    plt.figure(f"{channel} preselection efficiencies", figsize=(10, 10))
    plt.errorbar(bins_mean, prompt_preselection_efficiencies_array, xerr = bins_width, fmt = 'o', markersize=5, elinewidth = 2, capsize=4, label="Prompt")
    plt.errorbar(bins_mean, nonprompt_preselection_efficiencies_array, xerr = bins_width, fmt = 'o', markersize=5, elinewidth = 2, capsize=4, label="Nonprompt")
    plt.xlabel(r"$p_\mathrm{T}$ (GeV/$c$)")
    plt.ylabel("Preselection efficiency")
    plt.legend(loc='best')
    if save:
        plt.savefig(f"./{channel}_preselection_efficiency.png")
        plt.close("all")