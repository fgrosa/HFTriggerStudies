"""
Script for the preparation of the samples for the
training of ML models to be used in the HF triggers
"""

import os
import argparse
import uproot
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


def do_dca_smearing(df, nProng=None):
    """
    Method to do the DCA smearing for 2 prongs and 3 prongs

    Parameters
    -----------------
    - df: pandas dataframe containing all candidates with fFlagOrigin column
    - nProng: option to 2 prongs or 3 prongs

    Outputs
    -----------------
    - df: New dataframe with the smeared DCA columns
    """
    
    print("Start to do the smearing")
    # Open the input files
    file_names = ["sigmaDcaXY_LHC22q_pass2_LHCC.root", "sigmaDcaZ_LHC22q_pass2_LHCC.root"]
    input_files = [TFile.Open(name) for name in file_names]
    
    # Extract DCA resolution histograms
    dca_reso = {}
    for i, par in enumerate(["XY", "Z"]):
        gDca = input_files[i].Get("can")
        dca_reso[par] = gDca.GetPrimitive(f"tge_DCA_res_withPVrefit_all")
    
    # Add smeared DCA columns to the dataframe
    smear_cols = ["XY1", "XY2", "Z1", "Z2"]
    if nProng == 3:
        smear_cols.extend(["XY3", "Z3"])
    for col in smear_cols:
        dca_col = f"fDCAPrim{col}"
        pt_col = f"fPT{col[-1]}"
        df[f"{dca_col}_SMEAR"] = [
            gRandom.Gaus(dca, dca_reso[col[:-1]].Eval(pt) * 1e-4)
            for dca, pt in zip(df[dca_col], df[pt_col])
        ]
    
    # Close the input files
    for file in input_files:
        file.Close()

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
def main(input_dir, max_files=1000, downscale_bkg=1., force=False):
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with configs
    """

    input_files = []
    for subdir in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, subdir)):
            for subsubdir in os.listdir(os.path.join(input_dir, subdir)):
                if os.path.isdir(os.path.join(input_dir, subdir, subsubdir)):
                    for file in os.listdir(os.path.join(input_dir, subdir, subsubdir)):
                        if "AO2D.root" in file:
                            input_files.append(os.path.join(
                                input_dir, subdir, subsubdir, file))
                        elif os.path.isdir(os.path.join(input_dir, subdir, subsubdir, file)):
                            for file2 in os.listdir(os.path.join(
                                input_dir, subdir, subsubdir, file)):
                                if "AO2D.root" in file2:
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
                df_2p = uproot.concatenate(list_of_2p_df, library="pd")
                if args.dosmearing:
                    df_2p = do_dca_smearing(df_2p, 2)

                df_2p_prompt, df_2p_nonprompt, df_2p_bkg = divide_df_for_origin(
                    df_2p)
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
                df_3p = uproot.concatenate(list_of_3p_df, library="pd")
                if args.dosmearing:
                    df_3p = do_dca_smearing(df_3p, 3)

                for channel_3p in bits_3p:
                    flags = df_3p["fHFSelBit"].astype(
                        int) & 2**bits_3p[channel_3p]
                    df_channel_3p = df_3p[flags.astype("bool").to_numpy()]
                    df_3p_prompt, df_3p_nonprompt, df_3p_bkg = divide_df_for_origin(
                        df_channel_3p,
                        ["fFlagOrigin", "fChannel", "fHFSelBit"],
                        channel=channels_3p[channel_3p]
                    )
                    df_3p_bkg = df_3p_bkg.sample(
                        frac=downscale_bkg, random_state=42)
                    df_3p_prompt.to_parquet(
                        os.path.join(
                            indir, f"Prompt_{channel_3p}.parquet.gzip"),
                        compression="gzip"
                    )
                    df_3p_nonprompt.to_parquet(
                        os.path.join(
                            indir, f"Nonprompt_{channel_3p}.parquet.gzip"),
                        compression="gzip"
                    )
                    df_3p_bkg.to_parquet(
                        os.path.join(indir, f"Bkg_{channel_3p}.parquet.gzip"),
                        compression="gzip"
                    )

                df_3p = None

            bar_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("input_dir", metavar="text", default="AO2D/trains",
                        help="input directory with AO2D input files")
    parser.add_argument("--max_files", type=int, default=1000,
                        help="max input files to be processed")
    parser.add_argument("--downscale_bkg", type=float, default=1.,
                        help="fraction of bkg to be kept")
    parser.add_argument("--force", action="store_true", default=False,
                        help="force re-creation of output files")
    parser.add_argument("--dosmearing", action="store_true", default=False,
                        help="do smearing on the dca of daughter tracks ")
    args = parser.parse_args()

    main(args.input_dir, args.max_files, args.downscale_bkg, args.force)
