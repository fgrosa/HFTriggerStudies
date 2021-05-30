declare -a batch_sizes=( 100 1000 10000 100000 )
parallel -j4 python3 TestBonsaiBDT.py --batchSize {} ::: ${batch_sizes[@]}