#!/bin/bash

declare -a arrayOfFiles=("core_dump*" "localhost*")

for FILE in "${arrayOfFiles[@]}"
do
    rm $FILE 2>/dev/null
done
