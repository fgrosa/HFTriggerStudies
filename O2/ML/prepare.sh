#!/bin/bash

# \brief prepare the data samples for the training 

# list all the directories in training_samples/
outputDir="./training_samples/"
dataDirs=()
for dir in ${outputDir}*
do
    dataDirs+=($dir)
done

# if you want to enter a virtual environment
virtual_env=True
enter_virtual_env="source /home/abigot/Documents/ml_env/bin/activate"
if [ $virtual_env = True ]
then
    $enter_virtual_env
fi

# prepare command
for dir in ${dataDirs[@]}
do
    python3 prepare_samples.py $dir
done