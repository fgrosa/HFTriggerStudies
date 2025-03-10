# Code for BDT trainings to be used in HF software triggers
## Requirements
In order to execute the code in this folder, the following python libraries are needed:
- [hipe4ml](https://github.com/hipe4ml/hipe4ml)
- [hipe4ml_converter](https://github.com/hipe4ml/hipe4ml_converter)
- [alive_progress](https://github.com/rsalmei/alive-progress)

## Main steps
### Download training samples from hyperloop
In order to download the derived `AO2D.root` files produced on [hyperloop](https://alimonitor.cern.ch/hyperloop/), the following script can be used:
```python
python3 download_train_output.py input_files.txt output_directory
```
Where the list of input files is provided in the [input_files.txt](https://github.com/fgrosa/HFTriggerStudies/blob/main/O2/ML/input_files.txt). The list of input files can be directly copied from the hyperloop interface from by clicking on the train, `Submitted Jobs` &rarr; `Copy all output directories`.
The `output_directory` instead is the path to an existing directory where to download the `AO2D.root` files.

### Prepare the samples
In order to prepare the samples the following script can be used:
```python
python3 prepare_samples.py input_directory
```
Where `input directory` is the directory where the `AO2D.root` files have been downloaded from hyperloop.

### Perform training
In order to perform the training and produce the BDT models to be used in the triggers, the following script can be used:
```python
python3 train_hf_triggers.py config.yml
```
Where `config.yml` is a config file containing all the parameters about the data sample to be used, the channel, and the BDT parameters, such as [config_training_D0.yml](https://github.com/fgrosa/HFTriggerStudies/blob/main/O2/ML/config_training_D0.yml) for the D<sup>0</sup> meson or [config_training_Dplus.yml](https://github.com/fgrosa/HFTriggerStudies/blob/main/O2/ML/config_training_Dplus.yml) for the D<sup>+</sup> meson.

## Bash scripts
### Download
`download.sh` needs:
- `LHC22b1a_input_files.txt`, `LHC22b1b_input_files.txt` and `LHC21k6_pp_input_files.txt` as input files
- the corresponding train numbers that can be directly entered in `download.sh` or via a command line (thanks to the `INPUT` variable)

The files are then downloaded in dedicated directories inside `./training_samples/`.

### Prepare
`prepare.sh` browses the directories inside `./training_samples/` to prepare the data.

### Train
`train.sh` needs the particle's name in argument to modify the following entries in `config_training_particle.yml` (already existing):
- data files for `Prompt`, `Nonprompt` and `Bkg`
- `channel`
- `training_vars`
- `output: directory` 

 It then trains the BDT using this modified `.yml`.