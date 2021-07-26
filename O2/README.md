# O2
Run task for HF filters:
```sh
o2-analysis-hf-track-index-skims-creator -b --configuration json://dpl-config-triggerHF.json --resources-monitoring 2 | o2-analysis-hf-filter -b --configuration json://dpl-config-triggerHF.json --resources-monitoring 2 --fairmq-ipc-prefix .
```