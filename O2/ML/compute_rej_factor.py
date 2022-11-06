"""
Script for evaluation of rejection factor
"""

import os
import argparse
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import uproot
import yaml

PDGCHARM = {
    "Dplus": 411,
    "Dzero": 421,
    "Dstar": 413,
    "Ds": 431,
    "Lc": 4122
}

HADLABEL = {
    "Dplus": r"$\mathrm{D^+}$",
    "Dzero": r"$\mathrm{D^0}$",
    "Dstar": r"$\mathrm{D^{*+}}$",
    "Ds": r"$\mathrm{D_s^+}$",
    "Lc": r"$\Lambda_\mathrm{c}^+$"
}

BEAUTYDECAYLABEL = {
    "Dplus": r"$\mathrm{B^{0}\rightarrow D^-\pi^+}$",
    "Dzero": r"$\mathrm{B^{+}\rightarrow \overline{D}^0\pi^+}$",
    "Dstar": r"$\mathrm{B^{0}\rightarrow D^{*-}\pi^+}$",
    "Ds": r"$\mathrm{B_s^{0}\rightarrow D_s^-\pi^+}$",
    "Lc": r"$\Lambda_\mathrm{b}^{0}\rightarrow \Lambda_\mathrm{c}^+\pi^-$",
}

FEMTOLABEL = {
    "Dplus": r"$\mathrm{p}-\mathrm{D^+}$",
    "Dzero": r"$\mathrm{p}-\mathrm{D^0}$",
    "Ds": r"$\mathrm{p}-\mathrm{D_s^+}$",
    "Lc": r"$\mathrm{p}-\Lambda_\mathrm{c}^+$"
}


#pylint: disable=invalid-name
def scan_selections_beauty(df, dca_cuts, pt_cuts, def_cut_bkg):
    """
    Helper function to scan selections for beauty triggers
    """

    n_sel_coll_bkgcut, n_sel_coll_npcut = [], []
    for pt_cut in pt_cuts:
        df_pt = df.query(f"fPt > {pt_cut}")
        for dca_cut in dca_cuts:
            df_sel = df_pt.query(f"abs(fDCAXY) > {dca_cut}")
            n_sel_coll_bkgcut.append([])
            n_sel_coll_npcut.append([])
            for bkg_cut in np.arange(0., 1., 0.01):
                df_sel_bdt = df_sel.query(f"fBkgBDT < {bkg_cut}")
                n_sel_coll_bkgcut[-1].append(len(np.unique(df_sel_bdt["fCollisionIndex"].to_numpy())))

            df_sel_bdt = df_sel.query(f"fBkgBDT < {def_cut_bkg}")
            for np_cut in np.arange(0., 1., 0.01):
                df_sel_bdtnp = df_sel_bdt.query(f"fNonpromptBDT > {np_cut}")
                n_sel_coll_npcut[-1].append(len(np.unique(df_sel_bdtnp["fCollisionIndex"].to_numpy())))

            n_sel_coll_bkgcut[-1] = np.array(n_sel_coll_bkgcut[-1])
            n_sel_coll_npcut[-1] = np.array(n_sel_coll_npcut[-1])

    return n_sel_coll_bkgcut, n_sel_coll_npcut


#pylint: disable=invalid-name
def scan_selections_highpt(df, pt_cuts):
    """
    Helper function to scan selections for high pt triggers
    """

    n_sel_coll_bkgcut = []
    for pt_cut in pt_cuts:
        df_pt = df.query(f"fPt > {pt_cut}")
        n_sel_coll_bkgcut.append([])
        for bkg_cut in np.arange(0., 1., 0.01):
            df_sel_bdt = df_pt.query(f"fBkgBDT < {bkg_cut}")
            n_sel_coll_bkgcut[-1].append(len(np.unique(df_sel_bdt["fCollisionIndex"].to_numpy())))

        n_sel_coll_bkgcut[-1] = np.array(n_sel_coll_bkgcut[-1])

    return n_sel_coll_bkgcut


#pylint: disable=invalid-name
def scan_selections_femto(df, kstar_cuts, pt_cuts, def_cut_bkg):
    """
    Helper function to scan selections for femto triggers
    """

    n_sel_coll_bkgcut, n_sel_coll_pcut = [], []
    for pt_cut in pt_cuts:
        df_pt = df.query(f"fPt > {pt_cut}")
        for kstar_cut in kstar_cuts:
            df_sel = df_pt.query(f"fKStar < {kstar_cut}")
            n_sel_coll_bkgcut.append([])
            n_sel_coll_pcut.append([])
            for bkg_cut in np.arange(0., 1., 0.01):
                df_sel_bdt = df_sel.query(f"fBkgBDT < {bkg_cut}")
                n_sel_coll_bkgcut[-1].append(len(np.unique(df_sel_bdt["fCollisionIndex"].to_numpy())))

            df_sel_bdt = df_sel.query(f"fBkgBDT < {def_cut_bkg}")
            for p_cut in np.arange(0., 1., 0.01):
                df_sel_bdtp = df_sel_bdt.query(f"fPromptBDT > {p_cut}")
                n_sel_coll_pcut[-1].append(len(np.unique(df_sel_bdtp["fCollisionIndex"].to_numpy())))

            n_sel_coll_bkgcut[-1] = np.array(n_sel_coll_bkgcut[-1])
            n_sel_coll_pcut[-1] = np.array(n_sel_coll_pcut[-1])

    return n_sel_coll_bkgcut, n_sel_coll_pcut


#pylint: disable=too-many-locals, invalid-name
def scan_selections_doublecharm(df, pt_cuts, def_cut_bkg, def_cut_prompt):
    """
    Helper function to scan selections for double-charm triggers
    """

    n_sel_coll = [[], [], []]
    for pt_cut in product(*pt_cuts.values()):
        cut_string_2p = ""
        cut_string_3p = ""
        for ihad, had in enumerate(pt_cuts):
            if had == "Dzero":
                cut_string_2p += f"(fParticleID == {PDGCHARM[had]} and fBkgBDT < {def_cut_bkg[had]} and fPromptBDT > {def_cut_prompt[had]} and fPt > {pt_cut[ihad]})"
            else:
                cut_string_3p += f"(fParticleID == {PDGCHARM[had]} and fBkgBDT < {def_cut_bkg[had]} and fPromptBDT > {def_cut_prompt[had]} and fPt > {pt_cut[ihad]})"
                if ihad < len(pt_cut)-1:
                    cut_string_3p += " or "

        sel_2p = df.query(cut_string_2p)["fCollisionIndex"].to_numpy()
        sel_3p = df.query(cut_string_3p)["fCollisionIndex"].to_numpy()

        repeated_2p, repeated_3p = [], []
        for idx in sel_2p:
            if list(sel_2p).count(idx) > 1:
                repeated_2p.append(idx)
        for idx in sel_3p:
            if list(sel_3p).count(idx) > 1:
                repeated_3p.append(idx)

        n_sel_coll[0].append(len(np.unique(repeated_2p)))
        n_sel_coll[1].append(len(np.unique(repeated_3p)))
        n_sel_coll[2].append(len(list(set(sel_2p) & set(sel_3p))))

    for i in range(3):
        n_sel_coll[i] = np.array(n_sel_coll[i])

    return n_sel_coll


#pylint: disable=invalid-name
def apply_def_triggers(df_beauty, df_charm, df_femto, cfg):
    """
    Helper function to evaluate rejection factors for default triggers
    """

    n_sel_coll = {}
    sel_coll = []
    for had in PDGCHARM:
        had_to_sel = had
        if PDGCHARM[had] == 413:
            had_to_sel = "Dzero"

        # beauty trigger
        df_had_beauty = df_beauty.query(f"fParticleID == {PDGCHARM[had]} and "
                                        f"fBkgBDT < {cfg['default_selections'][had_to_sel]['bdt_background']} and "
                                        f"fNonpromptBDT > {cfg['default_selections'][had_to_sel]['bdt_nonprompt']} and "
                                        f"abs(fDCAXY) > {cfg['default_selections'][had_to_sel]['dca_beauty']} and "
                                        f"fPt > {cfg['default_selections'][had_to_sel]['minpt']}")

        sel_coll_beauty = np.unique(df_had_beauty["fCollisionIndex"].to_numpy())
        n_sel_coll[f"beauty_{had}"] = len(sel_coll_beauty)
        sel_coll += list(sel_coll_beauty)

        if PDGCHARM[had] == 413:
            continue

        # femto trigger
        df_had_femto = df_femto.query(f"fParticleID == {PDGCHARM[had]} and "
                                      f"fBkgBDT < {cfg['default_selections'][had_to_sel]['bdt_background']} and "
                                      f"fPt > {cfg['default_selections'][had_to_sel]['minpt']} and "
                                      f"fPromptBDT > {cfg['default_selections'][had_to_sel]['bdt_prompt']} and "
                                      f"fKStar < {cfg['default_selections'][had_to_sel]['kstar']}")
        sel_coll_femto = np.unique(df_had_femto["fCollisionIndex"].to_numpy())
        n_sel_coll[f"femto_{had}"] = len(sel_coll_femto)
        sel_coll += list(sel_coll_femto)

        # highpt trigger
        df_had_charm = df_charm.query(f"fParticleID == {PDGCHARM[had]} and "
                                      f"fBkgBDT < {cfg['default_selections'][had_to_sel]['bdt_background']} and "
                                      f"fPt > {cfg['default_selections'][had_to_sel]['minpt_highpt']}")
        sel_coll_highpt = np.unique(df_had_charm["fCollisionIndex"].to_numpy())
        n_sel_coll[f"highpt_{had}"] = len(sel_coll_highpt)
        sel_coll += list(sel_coll_highpt)

    # doublecharm trigger
    df_charm_2p = df_charm.query(f"fParticleID == {PDGCHARM['Dzero']} and "
                                 f"fBkgBDT < {cfg['default_selections']['Dzero']['bdt_background']} and "
                                 f"fPromptBDT > {cfg['default_selections']['Dzero']['bdt_prompt']} and "
                                 f"fPt > {cfg['default_selections']['Dzero']['minpt_doublecharm']}")

    sel_string_3p = ""
    for had in ["Dplus", "Ds", "Lc"]:
        if had == "Lc":
            sel_string_3p += f"(fParticleID == {PDGCHARM[had]} and fBkgBDT < {cfg['default_selections'][had]['bdt_background']} and fPromptBDT > {cfg['default_selections'][had]['bdt_prompt']} and fPt > {cfg['default_selections'][had]['minpt_doublecharm']})"
        else:
            sel_string_3p += f"(fParticleID == {PDGCHARM[had]} and fBkgBDT < {cfg['default_selections'][had]['bdt_background']} and fPromptBDT > {cfg['default_selections'][had]['bdt_prompt']} and fPt > {cfg['default_selections'][had]['minpt_doublecharm']}) or "

    df_charm_3p = df_charm.query(sel_string_3p)

    sel_2p = df_charm_2p["fCollisionIndex"].to_numpy()
    sel_3p = df_charm_3p["fCollisionIndex"].to_numpy()
    repeated_2p, repeated_3p = [], []
    for idx in sel_2p:
        if list(sel_2p).count(idx) > 1:
            repeated_2p.append(idx)
    for idx in sel_3p:
        if list(sel_3p).count(idx) > 1:
            repeated_3p.append(idx)

    sel_coll += list(np.unique(repeated_2p))
    sel_coll += list(np.unique(repeated_3p))
    sel_coll += list(set(sel_2p) & set(sel_3p))

    n_sel_coll["doublecharm_2prong"] = len(np.unique(repeated_2p))
    n_sel_coll["doublecharm_3prong"] = len(np.unique(repeated_3p))
    n_sel_coll["doublecharm_23prong"] = len(list(set(sel_2p) & set(sel_3p)))
    n_sel_coll["tot"] = len(np.unique(sel_coll))

    return n_sel_coll


def get_data_frames(input_dir):
    """
    Helper method to get pandas dataframes
    """

    df_beauty, df_charm, df_femto, df_names = ([] for _ in range(4))
    n_coll = []
    for subdir in os.listdir(input_dir):
        for subsubdir in os.listdir(os.path.join(input_dir, subdir)):
            for subsubsubdir in os.listdir(os.path.join(input_dir, subdir, subsubdir)):
                infile = uproot.open(
                    os.path.join(input_dir, subdir, subsubdir, subsubsubdir, "AO2D.root")
                )
                for directory in infile.keys():
                    if len(directory.split(sep="/")) == 1 and "DF" in directory.split(sep="/")[0]:
                        df_name = directory.split(sep="/")[0]
                        df_names.append(df_name)
                        trees = infile[df_name]
                        n_coll.append(len(trees["O2hfoptimtreecoll"].arrays(library="pd")))
                        df_beauty.append(trees["O2hfoptimtreeb"].arrays(library="pd"))
                        df_charm.append(trees["O2hfoptimtreec"].arrays(library="pd"))
                        df_femto.append(trees["O2hfoptimtreef"].arrays(library="pd"))

    return n_coll, df_beauty, df_charm, df_femto, df_names

#pylint: disable=too-many-locals,too-many-branches,too-many-statements
def compute_rej_factors(config_file):
    """
    Main function
    """

    with open(config_file, "r") as yml_cfg:  # pylint: disable=unspecified-encoding
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    doublecharm_cuts = {
        "Dzero": cfg["doublecharm"]["Dzero_pt_to_test"],
        "Dplus": cfg["doublecharm"]["Dplus_pt_to_test"],
        "Ds": cfg["doublecharm"]["Ds_pt_to_test"],
        "Lc": cfg["doublecharm"]["Lc_pt_to_test"]
    }
    doublecharm_bkgbdt_defcuts = {
        "Dzero": cfg["default_selections"]["Dzero"]["bdt_background"],
        "Dplus": cfg["default_selections"]["Dplus"]["bdt_background"],
        "Ds": cfg["default_selections"]["Ds"]["bdt_background"],
        "Lc": cfg["default_selections"]["Lc"]["bdt_background"]
    }
    doublecharm_pbdt_defcuts = {
        "Dzero": cfg["default_selections"]["Dzero"]["bdt_prompt"],
        "Dplus": cfg["default_selections"]["Dplus"]["bdt_prompt"],
        "Ds": cfg["default_selections"]["Ds"]["bdt_prompt"],
        "Lc": cfg["default_selections"]["Lc"]["bdt_prompt"]
    }

    n_coll_list, df_list_beauty, df_list_charm, df_list_femto, df_name_list = get_data_frames(cfg["input"])
    n_coll_tot = sum(n_coll_list)
    n_sel_coll_beauty_bkgcut, n_sel_coll_beauty_npcut = {}, {}
    n_sel_coll_highpt_bkgcut = {}
    n_sel_coll_femto_bkgcut, n_sel_coll_femto_pcut = {}, {}
    n_sel_coll_doublecharm = []
    with alive_bar(len(df_list_beauty), dual_line=True, title='Dataframes in AO2D') as bar_alive:
        for idf, (df_beauty, df_charm, df_femto, df_name) in enumerate(
            zip(df_list_beauty, df_list_charm, df_list_femto, df_name_list)):
            # if idf > 0:
            #     continue

            if cfg["doublecharm"]["active"]:
                # test selections for double charm triggers
                bar_alive.text = f'-> Processing double-charm trigger for dataframe: {df_name}'
                tmp_doublecharm = scan_selections_doublecharm(
                    df_charm,
                    doublecharm_cuts,
                    doublecharm_bkgbdt_defcuts,
                    doublecharm_pbdt_defcuts
                )
                if idf == 0:
                    n_sel_coll_doublecharm = tmp_doublecharm
                else:
                    for itrig, _ in enumerate(n_sel_coll_doublecharm):
                        n_sel_coll_doublecharm[itrig] += tmp_doublecharm[itrig]

            bar_alive.text = f'-> Processing other triggers for dataframe: {df_name}'
            for had in PDGCHARM:

                # test selections for beauty triggers
                if cfg["beauty"]["active"]:
                    df_beauty_part = df_beauty.query(f"fParticleID == {PDGCHARM[had]}")
                    if PDGCHARM[had] != 413:
                        had_for_sel = had
                    else:
                        had_for_sel = "Dzero"

                    tmp_bkgcut, tmp_npcut = scan_selections_beauty(
                        df_beauty_part,
                        cfg["beauty"]["dca_to_test"],
                        cfg["beauty"]["pt_to_test"],
                        cfg["default_selections"][had_for_sel]["bdt_background"]
                    )
                    if idf == 0:
                        n_sel_coll_beauty_bkgcut[had] = tmp_bkgcut
                        n_sel_coll_beauty_npcut[had] = tmp_npcut
                    else:
                        for icut, _ in enumerate(n_sel_coll_beauty_bkgcut[had]):
                            n_sel_coll_beauty_bkgcut[had][icut] += tmp_bkgcut[icut]
                            n_sel_coll_beauty_npcut[had][icut] += tmp_npcut[icut]

                if PDGCHARM[had] == 413:
                    continue

                # test selections for high-pT triggers
                if cfg["highpt"]["active"]:
                    df_charm_part = df_charm.query(f"fParticleID == {PDGCHARM[had]}")

                    tmp_bkgcut = scan_selections_highpt(
                        df_charm_part,
                        cfg["highpt"]["pt_to_test"],
                    )
                    if idf == 0:
                        n_sel_coll_highpt_bkgcut[had] = tmp_bkgcut
                    else:
                        for icut, _ in enumerate(n_sel_coll_highpt_bkgcut[had]):
                            n_sel_coll_highpt_bkgcut[had][icut] += tmp_bkgcut[icut]


                # test selections for femto triggers
                if cfg["femto"]["active"]:
                    df_femto_part = df_femto.query(f"fParticleID == {PDGCHARM[had]}")

                    tmp_bkgcut, tmp_pcut = scan_selections_femto(
                        df_femto_part,
                        cfg["femto"]["kstar_to_test"],
                        cfg["femto"]["pt_to_test"],
                        cfg["default_selections"][had_for_sel]["bdt_background"]
                    )
                    if idf == 0:
                        n_sel_coll_femto_bkgcut[had] = tmp_bkgcut
                        n_sel_coll_femto_pcut[had] = tmp_pcut
                    else:
                        for icut, _ in enumerate(n_sel_coll_femto_bkgcut[had]):
                            n_sel_coll_femto_bkgcut[had][icut] += tmp_bkgcut[icut]
                            n_sel_coll_femto_pcut[had][icut] += tmp_pcut[icut]

            # apply default trigger
            if idf == 0:
                n_sel_events_def = apply_def_triggers(df_beauty, df_charm, df_femto, cfg)
            else:
                n_sel_events_tmp = apply_def_triggers(df_beauty, df_charm, df_femto, cfg)
                for trigger in n_sel_events_def:
                    n_sel_events_def[trigger] += n_sel_events_tmp[trigger]

            bar_alive()

    # plots
    for had in PDGCHARM:
        if PDGCHARM[had] != 413:
            had_for_sel = had
        else:
            had_for_sel = "Dzero"

        # plot results for beauty
        if cfg["beauty"]["active"]:
            fig_beauty, ax_beauty = plt.subplots(1, 2, figsize=(16, 7))
            for ipt_cut, pt_cut in enumerate(cfg['beauty']['pt_to_test']):
                for idca_cut, dca_cut in enumerate(cfg['beauty']['dca_to_test']):
                    icut = (ipt_cut * len(cfg['beauty']['dca_to_test'])) + idca_cut
                    ax_beauty[0].plot(np.arange(0., 1., 0.01), n_sel_coll_beauty_bkgcut[had][icut]/n_coll_tot,
                                    marker='o', markersize=3,
                                    label=rf"$p_\mathrm{{T}}$({HADLABEL[had]}) > {pt_cut} GeV$/c$, "
                                    rf"DCAxy($\pi$) > {dca_cut*10000:.0f} $\mu$m")
                    ax_beauty[1].plot(np.arange(0., 1., 0.01), n_sel_coll_beauty_npcut[had][icut]/n_coll_tot,
                                    marker='o', markersize=3)

            ax_beauty[0].grid(True)
            ax_beauty[0].set_ylim(1.e-7,1.e-2)
            ax_beauty[0].set_title(f"{BEAUTYDECAYLABEL[had]}", fontsize=15)
            ax_beauty[0].set_yscale('log')
            ax_beauty[0].set_xlabel("Minimum threshold on background BDT", fontsize=15)
            ax_beauty[0].set_ylabel(
                r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
                fontsize=15)
            ax_beauty[0].tick_params(axis="x", labelsize=15)
            ax_beauty[0].tick_params(axis="y", labelsize=15)
            ax_beauty[0].legend(loc="best", fontsize=15)
            plt.tight_layout()

            ax_beauty[1].grid(True)
            ax_beauty[1].set_ylim(1.e-7,1.e-2)
            ax_beauty[1].set_yscale('log')
            ax_beauty[1].set_title(f"{BEAUTYDECAYLABEL[had]}", fontsize=15)
            ax_beauty[1].set_xlabel(f"Maximum threshold on non-prompt {HADLABEL[had]} BDT"
                                    f"(Background BDT < {cfg['default_selections'][had_for_sel]['bdt_background']})",
                                    fontsize=15)
            ax_beauty[1].set_ylabel(
                r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
                fontsize=15)
            ax_beauty[1].tick_params(axis="x", labelsize=15)
            ax_beauty[1].tick_params(axis="y", labelsize=15)
            plt.tight_layout()

            fig_beauty.savefig(os.path.join(cfg['output']['dir'], f"beauty_trigger_rejfact_{had}.pdf"))

        if PDGCHARM[had] == 413:
            continue

        # plot result for high-pt
        if cfg["highpt"]["active"]:
            fig_highpt, ax_highpt = plt.subplots(1, 1, figsize=(8, 7))
            for ipt_cut, pt_cut in enumerate(cfg['highpt']['pt_to_test']):
                ax_highpt.plot(np.arange(0., 1., 0.01), n_sel_coll_highpt_bkgcut[had][ipt_cut]/n_coll_tot,
                            marker='o', markersize=3,
                            label=rf"$p_\mathrm{{T}}$({HADLABEL[had]}) > {pt_cut} GeV$/c$")

            ax_highpt.grid(True)
            ax_highpt.set_ylim(1.e-7,1.e-2)
            ax_highpt.set_title(HADLABEL[had], fontsize=15)
            ax_highpt.set_yscale('log')
            ax_highpt.set_xlabel("Maximum threshold on background BDT", fontsize=15)
            ax_highpt.set_ylabel(
                r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
                fontsize=15)
            ax_highpt.tick_params(axis="x", labelsize=15)
            ax_highpt.tick_params(axis="y", labelsize=15)
            ax_highpt.legend(loc="best", fontsize=15)
            plt.tight_layout()

            fig_highpt.savefig(os.path.join(cfg['output']['dir'], f"highpt_trigger_rejfact_{had}.pdf"))

        # plot results for femto
        if cfg["femto"]["active"]:
            fig_femto, ax_femto = plt.subplots(1, 2, figsize=(16, 7))
            for ipt_cut, pt_cut in enumerate(cfg['femto']['pt_to_test']):
                for ikstar_cut, kstar_cut in enumerate(cfg['femto']['kstar_to_test']):
                    icut = (ipt_cut * len(cfg['femto']['kstar_to_test'])) + ikstar_cut
                    ax_femto[0].plot(np.arange(0., 1., 0.01), n_sel_coll_femto_bkgcut[had][icut]/n_coll_tot,
                                    marker='o', markersize=3,
                                    label=rf"$p_\mathrm{{T}}$({HADLABEL[had]}) > {pt_cut} GeV$/c$, "
                                    rf"$k^*$ < {kstar_cut*1000:.0f} MeV/$c$")
                    ax_femto[1].plot(np.arange(0., 1., 0.01), n_sel_coll_femto_pcut[had][icut]/n_coll_tot,
                                    marker='o', markersize=3)

            ax_femto[0].grid(True)
            ax_femto[0].set_ylim(1.e-7,1.e-2)
            ax_femto[0].set_title(f"{FEMTOLABEL[had]}", fontsize=15)
            ax_femto[0].set_yscale('log')
            ax_femto[0].set_xlabel("Maximum threshold on background BDT", fontsize=15)
            ax_femto[0].set_ylabel(
                r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
                fontsize=15)
            ax_femto[0].tick_params(axis="x", labelsize=15)
            ax_femto[0].tick_params(axis="y", labelsize=15)
            ax_femto[0].legend(loc="best", fontsize=15)
            plt.tight_layout()

            ax_femto[1].grid(True)
            ax_femto[1].set_ylim(1.e-7,1.e-2)
            ax_femto[1].set_yscale('log')
            ax_femto[1].set_title(f"{FEMTOLABEL[had]}", fontsize=15)
            ax_femto[1].set_xlabel(f"Minimum threshold on prompt {HADLABEL[had]} BDT"
                                f"(Background BDT < {cfg['default_selections'][had_for_sel]['bdt_background']})",
                                fontsize=15)
            ax_femto[1].set_ylabel(
                r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
                fontsize=15)
            ax_femto[1].tick_params(axis="x", labelsize=15)
            ax_femto[1].tick_params(axis="y", labelsize=15)
            plt.tight_layout()

            fig_femto.savefig(os.path.join(cfg['output']['dir'], f"femto_trigger_rejfact_{had}.pdf"))

    # plot result for double-charm
    if cfg["doublecharm"]["active"]:
        fig_doublecharm, ax_doublecharm = plt.subplots(1, 1, figsize=(15, 8))
        double_charm_labels = ["2-prong", "3-prong", "2-prong and 3-prong"]
        for itrig, _ in enumerate(n_sel_coll_doublecharm):
            ax_doublecharm.plot(range(len(n_sel_coll_doublecharm[itrig])),
                                n_sel_coll_doublecharm[itrig]/n_coll_tot,
                                marker='o', markersize=3, label=double_charm_labels[itrig])

        ax_doublecharm.grid(True)
        ax_doublecharm.set_ylim(1.e-7,1.e-2)
        ax_doublecharm.set_title("Collisions with multiple charm hadrons")
        ax_doublecharm.set_yscale('log')
        ax_doublecharm.set_ylabel(
            r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
            fontsize=15)

        ax_doublecharm.tick_params(axis="x", labelsize=15)
        ax_doublecharm.tick_params(axis="y", labelsize=15)
        x_titles_labels = []
        for cut in product(*doublecharm_cuts.values()):
            x_titles_labels.append(f"{cut[0]:.0f} {cut[1]:.0f} {cut[2]:.0f} {cut[3]:.0f}")
        ax_doublecharm.set_xticks(range(len(n_sel_coll_doublecharm[0])), x_titles_labels, rotation=90)
        ax_doublecharm.legend(loc="best", fontsize=15)
        plt.tight_layout()
        fig_doublecharm.savefig(os.path.join(cfg['output']['dir'], "doublecharm_trigger_rejfact.pdf"))

    # plot default
    fig_default, ax_default = plt.subplots(1, 1, figsize=(15, 8))
    ax_default.plot(range(len(n_sel_events_def)),
                    np.array(list(n_sel_events_def.values()))/n_coll_tot,
                    marker='o', markersize=3, color='black')
    ax_default.plot(range(len(n_sel_events_def)),
                    [1.e-4 for _ in range(len(n_sel_events_def))],
                    ls="--", color="red")
    ax_default.grid(True)
    ax_default.set_ylim(1.e-7,1.e-2)
    ax_default.set_title("HF triggers")
    ax_default.set_yscale('log')
    ax_default.set_ylabel(
        r"$N_\mathrm{collisions}^\mathrm{selected} / N_\mathrm{collisions}^\mathrm{analysed}$",
        fontsize=15)

    ax_default.tick_params(axis="x", labelsize=15)
    ax_default.tick_params(axis="y", labelsize=15)
    ax_default.set_xticks(range(len(n_sel_events_def)), n_sel_events_def.keys(), rotation=90)
    ax_default.legend(loc="best", fontsize=15)
    plt.tight_layout()
    fig_default.savefig(os.path.join(cfg['output']['dir'], "hf_trigger_rejfact.pdf"))

    plt.show()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Arguments")
    PARSER.add_argument("config", metavar="text",
                        default=".",
                        help="config_file")
    ARGS = PARSER.parse_args()

    compute_rej_factors(ARGS.config)
