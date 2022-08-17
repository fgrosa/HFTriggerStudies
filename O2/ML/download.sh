#!/bin/bash

# \brief code to download the data that will be used to train the BDT

# dataset=("beauty enriched" "charm enriched" "general purpose MC")
dataset=("LHC22b1a" "LHC22b1b" "LHC21k6_pp")

###################################################
# Step 1: Input files check
###################################################
# the input_files.txt must have been filled by hand for each dataset
inputFiles=("LHC22b1a_input_files.txt" "LHC22b1b_input_files.txt" "LHC21k6_pp_input_files.txt")
for data in ${dataset[@]}
do
    if [ ! -f ${data}"_input_files.txt" ] || [ ! -s ${data}"_input_files.txt" ]
    then
        echo ${data}"_input_files.txt does not exist or is empty"
        echo "Stopping the script..."
        exit 1
    fi
done

###################################################
# Step 2: Create download output directories
###################################################

# choice between entering the train numbers in this .sh file or via command line
INPUT=False
if [ $INPUT = True ]
then
    echo "Enter the Train number of LHC22b1a: "
    read b1aTrain
    echo "Enter the Train number of LHC22b1b: "
    read b1bTrain
    echo "Enter the Train number of LHC221k6_pp: "
    read k6Train
else
    b1aTrain="31002"
    b1bTrain="31003"
    k6Train="31202"
fi

# directories names
outputDir="./training_samples/"
b1aDir=$outputDir"LHC22b1a_train_"$b1aTrain
b1bDir=$outputDir"LHC22b1b_train_"$b1bTrain
k6Dir=$outputDir"LHC221k6_pp_train_"$k6Train
dataDirs=($b1aDir $b1bDir $k6Dir)
dirs=($outputDir $b1aDir $b1bDir $k6Dir)

# creating the directories (if they don't already exist)
for dir in ${dirs[@]}
do
    if [ ! -d $dir ]
    then
        mkdir $dir
    fi
done

###################################################
# Step 3: Download training samples from hyperloop
###################################################

# one needs to be connected to the Grid
echo "Are you connected to the Grid ? (y/n)"
read answer
if [ $answer != "y" ] && [ $answer != "Y" ]
then
    echo "Script interrupted, enter O2/O2Physics and connect to the Grid"
    exit 1
fi

# one needs to enter O2(Physics) to access the files to download
echo "Are you already in an O2(Physics) environment ? (y/n)"
read answer

if [ $answer = "y" ] || [ $answer = "Y" ] # if already in O2(Physics) then simple python command
then
    for i in $(seq 0 2)
    do
        echo " "
        echo "# Downloading files from" ${inputFiles[$i]}
        echo " "
        python3 download_train_output.py ${inputFiles[$i]} ${dataDirs[$i]}
        sleep 5
    done
else # if not in O2(Physics) then use setenv command to enter O2(Physics) just for this command line
    for i in $(seq 0 2)
    do
        echo " "
        echo "# Entering O2Physics/latest-master to download files from" ${inputFiles[$i]}
        echo " "
        alienv setenv O2Physics/latest-master-o2 -c python3 download_train_output.py ${inputFiles[$i]} ${dataDirs[$i]}
        sleep 5
    done
fi