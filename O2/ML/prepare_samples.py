"""
Script for the preparation of the samples for the
training of ML models to be used in the HF triggers
"""

import os
import argparse
import yaml
import uproot

# bits for 3 prongs
bits_3p = {"DplusToPiKPi": 0,
           "LcToPKPi": 1,
           "DsToKKPi": 2,
           "XicToPKPi": 3}

channels_3p = {"DplusToPiKPi": 1,
               "DsToKKPi": 2,
               "LcToPKPi": 3,
               "XicToPKPi": 4}

# function that divides a dataframe between prompt, non-prompt, bkg
def divide_df_for_origin(df, cols_to_remove=['fFlagOrigin'], channel=None):
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

    df_prompt = df.query("fFlagOrigin == 0")
    df_nonprompt = df.query("fFlagOrigin == 1")
    if channel is not None and "fChannel" in df_prompt.columns:
        df_prompt = df_prompt.query(f"fChannel == {channel}")
        df_nonprompt = df_nonprompt.query(f"fChannel == {channel}")
    df_bkg = df.query("fFlagOrigin == 2")
    cols_to_keep = set(df_prompt.columns)
    for col in cols_to_remove:
        cols_to_keep.remove(col)
    df_prompt = df_prompt[cols_to_keep]
    df_nonprompt = df_nonprompt[cols_to_keep]
    df_bkg = df_bkg[cols_to_keep]

    return df_prompt, df_nonprompt, df_bkg


def main(config): #pylint: disable=too-many-locals
    """
    Main function

    Parameters
    -----------------
    - config: dictionary with configs
    """

    input_files = cfg["input_files"]

    df_2p, df_3p = None, None
    for file in input_files:
        file_root = uproot.open(file)
        for tree_name in file_root.keys():
            if "O2hftrigtrain2p" in tree_name:
                if df_2p is None:
                    df_2p = file_root[tree_name].arrays(library="pd")
                else:
                    df_2p.append(file_root[tree_name].arrays(library="pd"))
            elif "O2hftrigtrain3p" in tree_name:
                if df_3p is None:
                    df_3p = file_root[tree_name].arrays(library="pd")
                else:
                    df_3p.append(file_root[tree_name].arrays(library="pd"))

        # 2-prongs --> only D0
        df_2p_prompt, df_2p_nonprompt, df_2p_bkg = divide_df_for_origin(df_2p)
        df_2p_prompt.to_parquet(
            os.path.join(os.path.split(file)[0], "Prompt_D0ToKPi.parquet.gzip"),
            compression="gzip"
        )
        df_2p_nonprompt.to_parquet(
            os.path.join(os.path.split(file)[0], "Nonprompt_D0ToKPi.parquet.gzip"),
                        compression="gzip"
        )
        df_2p_bkg.to_parquet(
            os.path.join(os.path.split(file)[0], "Bkg_D0ToKPi.parquet.gzip"),
            compression="gzip"
        )
        df_2p = None

        # 3-prongs --> D+, Ds+, Lc+, Xic+
        for channel_3p in bits_3p:
            flags = df_3p["fHFSelBit"].astype(int) & 2**bits_3p[channel_3p]
            df_channel_3p = df_3p[flags.astype("bool").to_numpy()]
            df_3p_prompt, df_3p_nonprompt, df_3p_bkg = divide_df_for_origin(
                df_channel_3p,
                ["fFlagOrigin", "fChannel", "fHFSelBit"],
                channel=channels_3p[channel_3p]
            )
            df_3p_prompt.to_parquet(
                os.path.join(os.path.split(file)[0], f"Prompt_{channel_3p}.parquet.gzip"),
                compression="gzip"
            )
            df_3p_nonprompt.to_parquet(
                os.path.join(os.path.split(file)[0], f"Nonprompt_{channel_3p}.parquet.gzip"),
                            compression="gzip"
            )
            df_3p_bkg.to_parquet(
                os.path.join(os.path.split(file)[0], f"Bkg_{channel_3p}.parquet.gzip"),
                compression="gzip"
            )

        df_3p = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text", default="input_files.yml",
                        help="config file with list of AO2D input files")
    args = parser.parse_args()

    with open(args.config, "r") as yml_cfg: #pylint: disable=unspecified-encoding
        cfg = yaml.load(yml_cfg, yaml.FullLoader)

    main(cfg)
