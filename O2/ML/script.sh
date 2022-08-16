outputDir="./training_samples/"
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

configFile="config_training_Dplus.yml"
sed -i 's,Prompt.*,'"Prompt: [$b1aDirs\, $b1bDirs]"',' $configFile
sed -i 's,Nonprompt.*,'"Nonprompt: [$b1aDirs\, $b1bDirs]"',' $configFile
sed -i 's,Bkg.*,'"Bkg: [$k6Dirs]"',' $configFile