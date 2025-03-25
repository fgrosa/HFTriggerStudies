"""
Script for the preparation of the samples for the
training of ML models to be used in the HF triggers
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
import yaml
from alive_progress import alive_bar
from ROOT import TFile, gRandom

# bits for 3 prongs
bits_3p = {"DplusToPiKPi": 0,
           "LcToPKPi": 1,
           "DsToKKPi": 2,
           "XicToPKPi": 3}

channels_3p = {"DplusToPiKPi": 1,
               "DsToKKPi": 2,
               "LcToPKPi": 3,
               "XicToPKPi": 4}


def apply_selections(df, is_signal, cfg, channel_3p = -1):
    """
    Method for application of preselections
    """

    if is_signal and cfg["require_good_coll"]:
        df.query("fIsInCorrectColl == 1", inplace=True)

    dcacut = cfg["preselection"]["dcaz"]

    if cfg["preselection"]["enable"]:
        if channel_3p == "D0ToPiKPi": # D0
            nsigma_cut_tpc = cfg["preselection"]["nsigma_tpc_kaonpion_d0"]
            nsigma_cut_tof = cfg["preselection"]["nsigma_tof_kaonpion_d0"]
            mass_cut_d0, mass_cut_d0bar = "", ""
            if cfg["bkg_from_data"] and not is_signal:
                mass_cut_d0 = "and (1.7 < fInvMassD0 < 1.8 or fInvMassD0 > 2.0)"
                mass_cut_d0bar = "and (1.7 < fInvMassD0bar < 1.8 or fInvMassD0bar > 2.0)"
            pid_sel_d0 = f"abs(fNsigmaPiTPC1) < {nsigma_cut_tpc} and (fNsigmaPiTOF1 != fNsigmaPiTOF1 or abs(fNsigmaPiTOF1) < {nsigma_cut_tof}) and abs(fNsigmaKaTPC2) < {nsigma_cut_tpc} and (fNsigmaKaTOF2 != fNsigmaKaTOF2 or abs(fNsigmaKaTOF2) < {nsigma_cut_tof})"
            pid_sel_d0bar = f"abs(fNsigmaPiTPC2) < {nsigma_cut_tpc} and (fNsigmaPiTOF2 != fNsigmaPiTOF2 or abs(fNsigmaPiTOF2) < {nsigma_cut_tof}) and abs(fNsigmaKaTPC1) < {nsigma_cut_tpc} and (fNsigmaKaTOF1 != fNsigmaKaTOF1 or abs(fNsigmaKaTOF1) < {nsigma_cut_tof})"
            string_sel = f"({pid_sel_d0} {mass_cut_d0}) or ({pid_sel_d0bar} {mass_cut_d0bar}) and abs(fDCAPrimZ1) < {dcacut} and abs(fDCAPrimZ2) < {dcacut}"
            df.query(string_sel, inplace=True)
        else:
            # common to all channels
            nsigma_cut_tpc = cfg["preselection"]["nsigma_tpc_kaon_3prong"]
            nsigma_cut_tof = cfg["preselection"]["nsigma_tof_kaon_3prong"]
            df.query(f"abs(fNsigmaKaTPC2) < {nsigma_cut_tpc} and (fNsigmaKaTOF2 != fNsigmaKaTOF2 or abs(fNsigmaKaTOF2) < {nsigma_cut_tof}) and abs(fDCAPrimZ1) < {dcacut} and abs(fDCAPrimZ2) < {dcacut} and abs(fDCAPrimZ3) < {dcacut}", inplace=True)

            if channel_3p == "DplusToPiKPi": # D+
                if cfg["bkg_from_data"] and not is_signal:
                    mass_cut_dplus = "1.7 < fInvMassDplus < 1.8 or 1.92 < fInvMassDplus < 1.98"
                    df.query(mass_cut_dplus, inplace=True)
  
            elif channel_3p == "LcToPKPi": # Lc
                nsigma_cut_tpc = cfg["preselection"]["nsigma_tpc_proton"]
                nsigma_cut_tof = cfg["preselection"]["nsigma_tof_proton"]
                mass_cut_pkpi, mass_cut_pikp = "", ""
                if cfg["bkg_from_data"] and not is_signal:
                    mass_cut_pkpi = "and (2.1 < fInvMassLcToPKPi < 2.2 or 2.4 < fInvMassLcToPKPi < 2.5)"
                    mass_cut_pikp = "and (2.1 < fInvMassLcToPiKP < 2.2 or 2.4 < fInvMassLcToPiKP < 2.5)"
                pid_sel_pkpi = f"abs(fNsigmaPrTPC1) < {nsigma_cut_tpc} and (fNsigmaPrTOF1 != fNsigmaPrTOF1 or abs(fNsigmaPrTOF1) < {nsigma_cut_tof})"
                pid_sel_pikp = f"abs(fNsigmaPrTPC3) < {nsigma_cut_tpc} and (fNsigmaPrTOF3 != fNsigmaPrTOF3 or abs(fNsigmaPrTOF3) < {nsigma_cut_tof})"
                string_sel = f"({pid_sel_pkpi} {mass_cut_pkpi}) or ({pid_sel_pikp} {mass_cut_pikp})"

                df.query(string_sel, inplace=True)

            elif channel_3p == "DsToKKPi": # Ds
                mass_cut_kkpi, mass_cut_pikk = "", ""
                deltamass_cut = cfg["preselection"]["delta_mass_kk"]
                if cfg["bkg_from_data"] and not is_signal:
                    mass_cut_kkpi = "and (1.7 < fInvMassDsToKKPi < 1.8 or fInvMassDsToKKPi > 2.05)"
                    mass_cut_pikk = "and (1.7 < fInvMassDsToPiKK < 1.8 or fInvMassDsToPiKK > 2.05)"
                deltamass_cut_kkpi = f"fDeltaMassKKFirst < {deltamass_cut}"
                deltamass_cut_pikk = f"fDeltaMassKKSecond < {deltamass_cut}"
                string_sel = f"({deltamass_cut_kkpi} {mass_cut_kkpi}) or ({deltamass_cut_pikk} {mass_cut_pikk})"

                df.query(string_sel, inplace=True)


def do_dca_smearing(df, n_prongs=None):
    """
    Method to do the DCA smearing for 2 prongs and 3 prongs

    Parameters
    -----------------
    - df: pandas dataframe containing all candidates with fFlagOrigin column
    - n_prongs: option to 2 prongs or 3 prongs

    Outputs
    -----------------
    - df: New dataframe with the smeared DCA columns
    """
    
    print("Start to do the smearing")
    # Open the input files
    file_names = ["sigmaDcaXY_LHC22q_pass2_LHCC.root", "sigmaDcaZ_LHC22q_pass2_LHCC.root"]
    input_files = [TFile.Open(name) for name in file_names]
    
    # Extract DCA resolution histograms
    dca_reso_data,  dca_reso_mc= {}, {}
    for i, par in enumerate(["XY", "Z"]):
        gDca = input_files[i].Get("can")
        dca_reso_data[par] = gDca.GetPrimitive("tge_DCA_res_withPVrefit_all_DATA")
        dca_reso_mc[par] = gDca.GetPrimitive("tge_DCA_res_withPVrefit_all_MC")

    # Add smeared DCA columns to the dataframe
    smear_cols = ["XY1", "XY2", "Z1", "Z2"]
    if n_prongs == 3:
        smear_cols.extend(["XY3", "Z3"])
    for col in smear_cols:
        dca_col = f"fDCAPrim{col}"
        pt_col = f"fPT{col[-1]}"
        df[f"{dca_col}_SMEAR"] = [
            gRandom.Gaus(dca, np.sqrt(dca_reso_data[col[:-1]].Eval(pt)**2 - dca_reso_mc[col[:-1]].Eval(pt)**2) * 1e-4)
            for dca, pt in zip(df[dca_col], df[pt_col])
        ]
    
    # Close the input files
    for file in input_files:
        file.Close()

    # Make a figure comparing the DCA variables before and after smearing
    num_cols = len(smear_cols)
    num_rows = num_cols // 2 + num_cols % 2
    _, axs = plt.subplots(num_rows, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    for i, col in enumerate(smear_cols):
        dca_col = f"fDCAPrim{col}"
        smear_col = f"{dca_col}_SMEAR"
        axs[i].hist(df[dca_col], bins=100, alpha=0.5, label="Before Smearing")
        axs[i].hist(df[smear_col], bins=100, alpha=0.5, label="After Smearing")
        axs[i].set_xlabel(col)
        axs[i].set_xlim(-1, 1)
        axs[i].set_yscale('log')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f"dca_comparison_{n_prongs}prong.png")
    #plt.show()
    return df


def divide_df_for_origin(df, cols_to_remove=None, channel=None):
    """
    Method to divide a dataframe in three (prompt, non-prompt, bkg)

    Parameters
    -----------------
    - df: pandas dataframe containing all candidates with fFlagOrigin column
    - cols_to_remove: columns to be removed from output dataframes
    - channel: integer corresponding a specific fChannel for signal

    Outputs
    -----------------
    - df_prompt: pandas dataframe containing only prompt signal
    - df_nonprompt: pandas dataframe containing only non-prompt signal
    - df_bkg: pandas dataframe containing only background candidates
    """

    if cols_to_remove is None:
        cols_to_remove = ['fFlagOrigin']

    df_prompt = df.query("fFlagOrigin == 1")
    df_nonprompt = df.query("fFlagOrigin == 2")
    if channel is not None and "fChannel" in df_prompt.columns:
        df_prompt = df_prompt.query(f"fChannel == {channel}")
        df_nonprompt = df_nonprompt.query(f"fChannel == {channel}")
    df_bkg = df.query("fFlagOrigin == 0")
    cols_to_keep = list(df_prompt.columns)
    for col in cols_to_remove:
        cols_to_keep.remove(col)
    df_prompt = df_prompt[cols_to_keep]
    df_nonprompt = df_nonprompt[cols_to_keep]
    df_bkg = df_bkg[cols_to_keep]

    return df_prompt, df_nonprompt, df_bkg


# pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks
def main(input_dir, config_file, max_files=1000, downscale_bkg=1., force=False, do_smearing=False):
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with configs
    """

    with open(config_file, "r") as yml_cfg:  # pylint: disable=bad-option-value
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    input_files = []
    for subdir in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, subdir)):
            for subsubdir in os.listdir(os.path.join(input_dir, subdir)):
                if os.path.isdir(os.path.join(input_dir, subdir, subsubdir)):
                    for file in os.listdir(os.path.join(input_dir, subdir, subsubdir)):
                        if "AO2D" in file:
                            input_files.append(os.path.join(
                                input_dir, subdir, subsubdir, file))
                        elif os.path.isdir(os.path.join(input_dir, subdir, subsubdir, file)):
                            for file2 in os.listdir(os.path.join(
                                input_dir, subdir, subsubdir, file)):
                                if "AO2D" in file2:
                                    input_files.append(os.path.join(
                                        input_dir, subdir, subsubdir, file, file2))

    df_2p, df_3p = None, None
    with alive_bar(len(input_files[:max_files])) as bar_alive:
        for file in input_files[:max_files]:
            print(f"\033[32mExtracting dataframes from input "
                  f"{file}\033[0m")

            file_root = uproot.open(file)
            indir = os.path.split(file)[0]

            # 2-prongs --> only D0
            is_d0_filtered = False
            for exfile in os.listdir(indir):
                if "D0ToKPi.parquet.gzip" in exfile:
                    is_d0_filtered = True
                    break
            if not is_d0_filtered or force:
                list_of_2p_df = []
                for tree_name in file_root.keys():
                    if "O2hftrigtrain2p" in tree_name:
                        list_of_2p_df.append(f"{file}:{tree_name}")
                if len(list_of_2p_df) > 0:
                    df_2p = uproot.concatenate(list_of_2p_df, library="pd")
                    if do_smearing:
                        df_2p = do_dca_smearing(df_2p, 2)

                    df_2p["fNsigmaPiTOF1"] = df_2p["fNsigmaPiTOF1"].replace(-999., np.nan)
                    df_2p["fNsigmaPiTOF2"] = df_2p["fNsigmaPiTOF2"].replace(-999., np.nan)
                    df_2p["fNsigmaKaTOF1"] = df_2p["fNsigmaKaTOF1"].replace(-999., np.nan)
                    df_2p["fNsigmaKaTOF2"] = df_2p["fNsigmaKaTOF2"].replace(-999., np.nan)

                    df_2p_prompt, df_2p_nonprompt, df_2p_bkg = divide_df_for_origin(df_2p)
                    apply_selections(df_2p_prompt, True, cfg, "D0ToPiKPi")
                    apply_selections(df_2p_nonprompt, True, cfg, "D0ToPiKPi")
                    apply_selections(df_2p_bkg, False, cfg, "D0ToPiKPi")
                    df_2p_bkg = df_2p_bkg.sample(
                        frac=downscale_bkg, random_state=42)
                    df_2p_prompt.to_parquet(
                        os.path.join(indir, "Prompt_D0ToKPi.parquet.gzip"),
                        compression="gzip"
                    )
                    df_2p_nonprompt.to_parquet(
                        os.path.join(indir, "Nonprompt_D0ToKPi.parquet.gzip"),
                        compression="gzip"
                    )
                    df_2p_bkg.to_parquet(
                        os.path.join(indir, "Bkg_D0ToKPi.parquet.gzip"),
                        compression="gzip"
                    )
                    df_2p = None

            # 3-prongs --> D+, Ds+, Lc+, Xic+
            is_3p_filtered = False
            for channel_3p in bits_3p:
                for exfile in os.listdir(indir):
                    if f"{channel_3p}.parquet.gzip" in exfile:
                        is_3p_filtered = True
                        break

            list_of_3p_df = []
            if not is_3p_filtered or force:
                for tree_name in file_root.keys():
                    if "O2hftrigtrain3p" in tree_name:
                        list_of_3p_df.append(f"{file}:{tree_name}")
                if len(list_of_3p_df) > 0:
                    df_3p = uproot.concatenate(list_of_3p_df, library="pd")
                    if do_smearing:
                        df_3p = do_dca_smearing(df_3p, 3)

                    df_3p["fNsigmaPiTOF1"] = df_3p["fNsigmaPiTOF1"].replace(-999., np.nan)
                    df_3p["fNsigmaPiTOF2"] = df_3p["fNsigmaPiTOF2"].replace(-999., np.nan)
                    df_3p["fNsigmaPiTOF3"] = df_3p["fNsigmaPiTOF3"].replace(-999., np.nan)
                    df_3p["fNsigmaPrTOF1"] = df_3p["fNsigmaPrTOF1"].replace(-999., np.nan)
                    df_3p["fNsigmaPrTOF2"] = df_3p["fNsigmaPrTOF2"].replace(-999., np.nan)
                    df_3p["fNsigmaPrTOF3"] = df_3p["fNsigmaPrTOF3"].replace(-999., np.nan)
                    df_3p["fNsigmaKaTOF1"] = df_3p["fNsigmaKaTOF1"].replace(-999., np.nan)
                    df_3p["fNsigmaKaTOF2"] = df_3p["fNsigmaKaTOF2"].replace(-999., np.nan)
                    df_3p["fNsigmaKaTOF3"] = df_3p["fNsigmaKaTOF3"].replace(-999., np.nan)

                    for channel_3p in bits_3p:
                        flags = df_3p["fHFSelBit"].astype(int) & 2**bits_3p[channel_3p]
                        df_channel_3p = df_3p[flags.astype("bool").to_numpy()]
                        df_3p_prompt, df_3p_nonprompt, df_3p_bkg = divide_df_for_origin(
                            df_channel_3p,
                            ["fFlagOrigin", "fChannel", "fHFSelBit"],
                            channel=channels_3p[channel_3p]
                        )
                        apply_selections(df_3p_prompt, True, cfg, channel_3p)
                        apply_selections(df_3p_nonprompt, True, cfg, channel_3p)
                        apply_selections(df_3p_bkg, False, cfg, channel_3p)

                        df_3p_bkg = df_3p_bkg.sample(frac=downscale_bkg, random_state=42)
                        df_3p_prompt.to_parquet(
                            os.path.join(indir, f"Prompt_{channel_3p}.parquet.gzip"),
                            compression="gzip"
                        )

                        with uproot.recreate(os.path.join(indir, f"Prompt_{channel_3p}.root")) as outfile:
                            outfile[f"tree{channel_3p}"] = df_3p_prompt
                        df_3p_nonprompt.to_parquet(
                            os.path.join(indir, f"Nonprompt_{channel_3p}.parquet.gzip"),
                            compression="gzip"
                        )
                        with uproot.recreate(os.path.join(indir, f"Nonprompt_{channel_3p}.root")) as outfile:
                            outfile[f"tree{channel_3p}"] = df_3p_nonprompt
                        df_3p_bkg.to_parquet(
                            os.path.join(indir, f"Bkg_{channel_3p}.parquet.gzip"),
                            compression="gzip"
                        )
                        with uproot.recreate(os.path.join(indir, f"Bkg_{channel_3p}.root")) as outfile:
                            outfile[f"tree{channel_3p}"] = df_3p_bkg

                    df_3p = None

            bar_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("input_dir", metavar="text", default="AO2D/trains",
                        help="input directory with AO2D input files")
    parser.add_argument("--config_file", metavar="text", default="config_prepare_samples.yml",
                        help="input yaml config file")
    parser.add_argument("--max_files", type=int, default=1000,
                        help="max input files to be processed")
    parser.add_argument("--downscale_bkg", type=float, default=1.,
                        help="fraction of bkg to be kept")
    parser.add_argument("--force", action="store_true", default=False,
                        help="force re-creation of output files")
    parser.add_argument("--dosmearing", action="store_true", default=False,
                        help="do smearing on the dca of daughter tracks ")
    args = parser.parse_args()

    main(args.input_dir, args.config_file, args.max_files,
         args.downscale_bkg, args.force, args.dosmearing)
