#!/bin/bash

# \brief modify config.yml file and train BDT
# \argument: Particle

Particle=$1

###################################################
# Step 1: Check the particle's name
###################################################
Particles=(D0 Dplus Ds Lc Xic)
ParticleInList=False
for particle in ${Particles[@]}
do
    if [ $Particle = $particle ]; then
        ParticleInList=True
    fi
done

if [ $ParticleInList = False ]; then
    echo $Particle "not in the particles list:" ${Particles[@]}
    exit 1
fi

###################################################
# Step 2: Prepare the directories' names
###################################################

outputDir="training_samples/"
dataset=("b1a" "b1b" "k6")
# path to the directories
for data in ${dataset[@]}
do
    ls $outputDir | grep "${data}" > $data".txt"
    sed -i 's,^,'"$outputDir"',' $data".txt"
    readarray -t ${data}Dir < $data".txt"
done

# separate the directories names with a ","
for dir in ${b1aDir[@]}; do b1aDirs+="$dir""\, "; done
for dir in ${b1bDir[@]}; do b1bDirs+="$dir""\, "; done
for dir in ${k6Dir[@]}; do k6Dirs+="$dir""\, "; done

# remove the "," on the last directory
b1aDirs=$(echo $b1aDirs | rev | cut -c3- | rev)
b1bDirs=$(echo $b1bDirs | rev | cut -c3- | rev)
k6Dirs=$(echo $k6Dirs | rev | cut -c3- | rev)
# we now have an array containing the directories'names for b1a, b1b and k6 data separately
# (might be useful in the future if several data directories of same "data type" are used for the same training)

###################################################
# Step 3: Config file check
###################################################
configFile="config_training_"$Particle".yml"
if [ ! -f $configFile ] || [ ! -s $configFile ]
then
    echo $configFile "does not exist or is empty"
    exit 1
fi

###################################################
# Step 4: Modify the config file
###################################################
# change the dirs
sed -i 's,Prompt.*,'"Prompt: [$b1aDirs\, $b1bDirs]"',' $configFile
sed -i 's,Nonprompt.*,'"Nonprompt: [$b1aDirs\, $b1bDirs]"',' $configFile
sed -i 's,Bkg.*,'"Bkg: [$k6Dirs]"',' $configFile

# change the decay channel
channel_options="  # options: D0ToKPi\, DplusToPiKPi\, DsToKKPi\, LcToPKPi\, XicToPKPi"
channels=(D0ToKPi DplusToPiKPi DsToKKPi LcToPKPi XicToPKPi)
if [ $Particle = D0 ]; then
    channel=D0ToKPi
elif [ $Particle = Dplus ]; then
    channel=DplusToPiKPi
elif [ $Particle = Ds ]; then
    channel=DsToKKPi
elif [ $Particle = Lc ]; then
    channel=LcToPKPi
elif [ $Particle = Xic ]; then
    channel=XicToPKPi
fi

sed -i 's,channel.*,'"channel: $channel$channel_options"',' $configFile

# change the training variables according to the number of prongs
twoProng_training_vars="[fPT1\, fDCAPrimXY1\, fDCAPrimZ1\, fPT2\, fDCAPrimXY2\, fDCAPrimZ2]"
threeProng_training_vars="[fPT1\, fDCAPrimXY1\, fDCAPrimZ1\, fPT2\, fDCAPrimXY2\, fDCAPrimZ2\, fPT3\, fDCAPrimXY3\, fDCAPrimZ3]"
if [ $Particle = D0 ]
then
    sed -i 's,training_vars.*,'"training_vars: $twoProng_training_vars"',' $configFile
else
    sed -i 's,training_vars.*,'"training_vars: $threeProng_training_vars"',' $configFile
fi


# change directory
sed -i 's,directory.*,'"directory: trainings/$Particle"',' $configFile

###################################################
# Step 4: Train the BDT
###################################################

# if you want to enter a virtual environment
virtual_env=False
enter_virtual_env="source /home/abigot/Documents/ml_env/bin/activate"
if [ $virtual_env = True ]
then
    $enter_virtual_env
fi

python3 train_hf_triggers.py $configFile
