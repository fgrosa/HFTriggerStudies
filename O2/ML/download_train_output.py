"""
Script for the download of files from hyperloop
"""

import os
import argparse
from ROOT import TGrid #pylint: disable=no-name-in-module

# main function
def main(infile, outpath):
    """
    Main function

    Parameters
    -----------------
    - infile: input file with list of directories separated by ','
    - outpath: output path
    """

    with open(infile) as f_txt: #pylint: disable=unspecified-encoding
        contents = f_txt.read()
    list_of_dirs = contents.split(",")

    grid = TGrid.Connect("alien://")

    for i_dir, indir in enumerate(list_of_dirs):
        jobdir = indir.split("/")[-1]
        outdir = os.path.join(outpath, jobdir)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        list_of_subdirs = grid.Ls(indir.replace("alien://", ""))
        print(f"download directory {i_dir+1}/{len(list_of_dirs)}")
        for i_sub, _ in enumerate(list_of_subdirs):
            if list_of_subdirs.GetFileName(i_sub).isdigit():
                subdir = list_of_subdirs.GetFileName(i_sub)
                os.system("alien_cp -y 2 -T 32 -name contain_root "
                          f"{indir}/{subdir} file://{outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("infile", metavar="text", default="list.txt",
                        help="list of directories with input files")
    parser.add_argument("outpath", metavar="text", default=".",
                        help="output path")
    args = parser.parse_args()

    main(args.infile, args.outpath)
