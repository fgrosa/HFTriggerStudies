# O2
Run task for HF filters:
```sh
o2-analysis-hf-track-index-skims-creator -b --configuration json://dpl-config-triggerHF.json | o2-analysis-hf-filter -b --configuration json://dpl-config-triggerHF.json --fairmq-ipc-prefix .
```