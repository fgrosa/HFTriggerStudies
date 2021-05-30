#!/usr/bin/python3

'''
Script for plotting results of BDT application benchmark 
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt

xgbVer = '1.3.3'
batchSizes = [100, 1000, 10000]
featuresForTrain = ['fd0MinDau', 'fDecayLength', 'fImpParProd', 'fCosP']
colors = ['forestgreen', 'lightseagreen', 'teal', 'steelblue', 'navy']

dfBatchSizes, figTime, timeVsBatchSize = {}, {}, {}
for iDf, batchSize in enumerate(batchSizes):
    dfBatchSizes[batchSize] = pd.read_parquet(f'outputs/timetests/timeBench_XGBoost_v{xgbVer}_batchSize{batchSize}'
                                              f'_features_{"-".join(featuresForTrain)}.parquet.gzip')

    nBins, clfNames = [], []
    for name in dfBatchSizes[batchSize].columns:
        if 'BBDT' in name:
            clfNames.append(name)
            nBins.append(float(name.strip('BBDT')))
        if name != 'n_estimators':
            if name not in timeVsBatchSize:
                timeVsBatchSize[name] = []
            timeVsBatchSize[name].append(dfBatchSizes[batchSize].query('n_estimators == 1500')[name].to_numpy()[0])

    figTime[batchSize] = plt.figure(figsize=(15, 8))
    plt.grid(True)
    plt.plot(dfBatchSizes[batchSize]['n_estimators'].to_numpy(), dfBatchSizes[batchSize]['XGBoost'].to_numpy(),
             label='xgboost', color='darkred')
    plt.plot(dfBatchSizes[batchSize]['n_estimators'].to_numpy(), dfBatchSizes[batchSize]['Treelite'].to_numpy(),
             label='treelite', color='chocolate')
    for iBin, (nBin, name) in enumerate(zip(nBins, clfNames)):
        plt.plot(dfBatchSizes[batchSize]['n_estimators'].to_numpy(), dfBatchSizes[batchSize][name].to_numpy(),
                 label=f'BBDT nbins = {nBin}', color=colors[iBin])
    plt.yscale('log')
    plt.legend(loc='best')
    plt.ylabel('time / candidate', size=15)
    plt.xlabel('n_estimators', size=15)
    figTime[batchSize].savefig(f'plots/Time_benchmark_batchSize{batchSize}.pdf')
    plt.close('all')

figTimeVsBatchSize = plt.figure(figsize=(15, 8))
plt.grid(True)
plt.plot(batchSizes, timeVsBatchSize['XGBoost'], label='xgboost', color='darkred')
plt.plot(batchSizes, timeVsBatchSize['Treelite'], label='treelite', color='chocolate')
for iBin, (nBin, name) in enumerate(zip(nBins, clfNames)):
    plt.plot(batchSizes, timeVsBatchSize[name], label=f'BBDT nbins = {nBin}', color=colors[iBin])
plt.yscale('log')
plt.xscale('log')
plt.legend(loc='best')
plt.ylabel('time / candidate', size=15)
plt.xlabel('batch size', size=15)
figTimeVsBatchSize.savefig(f'plots/Time_benchmark_vs_batchSize.pdf')
plt.close('all')

plt.show()
