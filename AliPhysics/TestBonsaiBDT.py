#!/usr/bin/python3

'''
Script for benchmarks of time performances of different BDT applications 
'''

import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import treelite
import treelite_runtime
from hep_ml.speedup import LookupClassifier
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml import plot_utils

parser = argparse.ArgumentParser(description='Arguments to pass')
parser.add_argument('--batchSize', type=int, default=1000000000,
                    help='batch size')
args = parser.parse_args()

#**************************************************************************
# Prepare the datasets
promptH = TreeHandler(['outputs/LHC20f4/a/Reco_prompt_D0.parquet_0.gzip',
                       'outputs/LHC20f4/a/Reco_prompt_D0.parquet_1.gzip'])
fdH = TreeHandler(['outputs/LHC20f4/a/Reco_FD_D0.parquet_0.gzip',
                   'outputs/LHC20f4/a/Reco_FD_D0.parquet_1.gzip'])
bkgH = TreeHandler(['outputs/ppGenPurpose/2018/Reco_bkg_D0.parquet_0.gzip',
                    'outputs/ppGenPurpose/2018/Reco_bkg_D0.parquet_1.gzip'])
promptH.apply_preselections('abs(fImpParProd) < 0.01')
fdH.apply_preselections('abs(fImpParProd) < 0.01')
bkgH.apply_preselections('abs(fImpParProd) < 0.01')

trainTestData = train_test_generator([promptH, bkgH], [1,0], test_size=0.5, random_state=42)
nCandTrain = len(trainTestData[0])
nCandTest = len(trainTestData[2])

varsToDraw = ['fPt', 'fInvMass', 'fd0MinDau', 'fDecayLength', 'fImpParProd', 'fCosP']
legLabels = ['background', r'prompt D$^0$']

figDistr = plot_utils.plot_distr([bkgH, promptH], varsToDraw, bins=100, labels=legLabels,
                                 log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
plt.savefig(f'plots/Vars_{"-".join(varsToDraw)}.pdf')
figCorr = plot_utils.plot_corr([bkgH, promptH], varsToDraw, legLabels)
plt.savefig(f'plots/VarsCorr_{"-".join(varsToDraw)}.pdf')
plt.close('all')

featuresForTrain = ['fd0MinDau', 'fDecayLength', 'fImpParProd', 'fCosP']
timeAppl, yPredTrain, yPredTest, yPredAppl = ({} for _ in range(4))

nEstimList = [100, 200, 300, 500, 750, 1000, 1500]
nBinsList = [2, 5, 10, 50, 100]
colors = ['forestgreen', 'lightseagreen', 'teal', 'steelblue', 'navy']

for iEstim, nEstim in enumerate(nEstimList):
    #**************************************************************************
    # Do the trainings
    # XGBoost (via hipe4ml)
    hypParam = {'max_depth': 4, 'n_estimators': nEstim, 'tree_method': 'hist', 'njobs': 4}
    xgbName = f'xgboost/XGBoost_v{xgb.__version__}_n_estimators{nEstim}_features_{"-".join(featuresForTrain)}.model'
    hdlName = f'xgboost/ModelHandler_XGBoost_v{xgb.__version__}_n_estimators{nEstim}_features_{"-".join(featuresForTrain)}.pickle'
    if os.path.isfile(hdlName):
        modelHdl = ModelHandler()
        modelHdl.load_model_handler(hdlName)
        modelClf = modelHdl.get_original_model()
    else:
        modelClf = xgb.XGBClassifier()
        modelHdl = ModelHandler(modelClf, featuresForTrain, hypParam)
        modelHdl.train_test_model(trainTestData)
        modelHdl.dump_original_model(xgbName, True)
        modelHdl.dump_model_handler(hdlName)
    if iEstim == 0:
        timeAppl['XGBoost'] = []

    # Bonsai BDT
    bBDT = []
    for nBins in nBinsList:
        maxCells = nBins**4
        bBDT.append(LookupClassifier(base_estimator=modelClf, n_bins=nBins, max_cells=maxCells))
        bBDT[-1].fit(trainTestData[0][featuresForTrain], trainTestData[1], sample_weight=None)
        if iEstim == 0:
            timeAppl[f'BBDT{nBins}'] = []

    # Treelite
    modelTreeLite = treelite.Model.load(xgbName, model_format='xgboost')
    libFile = f'treelite/XGBoost_v{xgb.__version__}_n_estimators{nEstim}_features_{"-".join(featuresForTrain)}.so'
    if not os.path.isfile(libFile):
        modelTreeLite.export_lib(toolchain='gcc', libpath=libFile, params={'parallel_comp': 32}, verbose=True)
    predictorTreeLite = treelite_runtime.Predictor(libFile, verbose=True)
    dmatTrain = treelite_runtime.DMatrix(trainTestData[0][featuresForTrain], feature_names=featuresForTrain)
    dmatTest = treelite_runtime.DMatrix(trainTestData[2][featuresForTrain], feature_names=featuresForTrain)
    if iEstim == 0:
        timeAppl['Treelite'] = []

    # RBDT
    #**************************************************************************
    # Benchmark results
    applSample = trainTestData[2][featuresForTrain].iloc[:args.batchSize]
    dmatAppl = treelite_runtime.DMatrix(applSample, feature_names=featuresForTrain)
    nCandApply = args.batchSize if args.batchSize < nCandTest else nCandTest

    # XGBoost
    yPredTrain['XGBoost'] = modelHdl.predict(trainTestData[0], False) # for ROC and residuals
    yPredTest['XGBoost'] = modelHdl.predict(trainTestData[2], False) # for ROC and residuals
    start = time.time()
    yPredAppl['XGBoost'] = modelHdl.predict(applSample, False)
    end = time.time()
    timeAppl['XGBoost'].append((end-start)/nCandApply)

    # Bonsai BDT
    for classif, nBins in zip(bBDT, nBinsList):
        yPredTrain[f'BBDT{nBins}'] = classif.predict_proba(trainTestData[0][featuresForTrain]) # for ROC and residuals
        yPredTest[f'BBDT{nBins}'] = classif.predict_proba(trainTestData[2][featuresForTrain]) # for ROC and residuals
        start = time.time()
        yPredAppl[f'BBDT{nBins}'] = classif.predict_proba(applSample)
        end = time.time()
        timeAppl[f'BBDT{nBins}'].append((end-start)/nCandApply)

    # Treelite
    yPredTrain['Treelite'] = predictorTreeLite.predict(dmatTrain) # for ROC and residuals
    yPredTest['Treelite'] = predictorTreeLite.predict(dmatTest) # for ROC and residuals
    start = time.time()
    yPredAppl['Treelite'] = predictorTreeLite.predict(dmatAppl)
    end = time.time()
    timeAppl['Treelite'].append((end-start)/nCandApply)

    #**************************************************************************
    # Some nice plots
    # ROC
    rocTrainTestFig = plot_utils.plot_roc_train_test(trainTestData[3], yPredTest['XGBoost'],
                                                     trainTestData[1], yPredTrain['XGBoost'],
                                                     None, legLabels)
    rocTrainTestFig.savefig(f'plots/ROC_XGboost_n_estimators_{nEstim}.pdf')
    pickle.dump(rocTrainTestFig, open(f'plots/ROC_XGboost_n_estimators_{nEstim}.pkl', 'wb'))
    for iBins, nBins in enumerate(nBinsList):
        rocTrainTestFig = plot_utils.plot_roc_train_test(trainTestData[3], yPredTest[f'BBDT{nBins}'][:, 1],
                                                         trainTestData[1], yPredTrain[f'BBDT{nBins}'][:, 1],
                                                         None, legLabels)
        rocTrainTestFig.savefig(f'plots/ROC_BBDT_nBins{nBins}_n_estimators_{nEstim}.pdf')
        pickle.dump(rocTrainTestFig, open(f'plots/ROC_BBDT_nBins{nBins}_n_estimators_{nEstim}.pkl', 'wb'))
    rocTrainTestFig = plot_utils.plot_roc_train_test(trainTestData[3], yPredTest['Treelite'],
                                                     trainTestData[1], yPredTrain['Treelite'],
                                                     None, legLabels)
    rocTrainTestFig.savefig(f'plots/ROC_Treelite_n_estimators_{nEstim}.pdf')
    pickle.dump(rocTrainTestFig, open(f'plots/ROC_Treelite_n_estimators_{nEstim}.pkl', 'wb'))

    # Score residuals
    resFig = plt.figure(figsize=(8, 8))
    for iBins, nBins in enumerate(nBinsList):
        plt.hist(yPredTest[f'BBDT{nBins}'][:, 1]-yPredTest['XGBoost'], bins=1000, histtype='step',
                 stacked=True, fill=False, label=f'BBDT nbins = {nBins}', color = colors[iBins])
    plt.hist(yPredTest['Treelite']-yPredTest['XGBoost'], bins=1000, histtype='step',
             stacked=True, fill=False, label='treelite', color='chocolate')
    plt.ylabel('entries', size=15)
    plt.xlabel('BDT output residual to XGBoost', size=15)
    plt.legend(loc='best')
    resFig.savefig(f'plots/Residuals_n_estimators_{nEstim}_batchSize{nCandApply}_lin.pdf')
    plt.yscale('log')
    resFig.savefig(f'plots/Residuals_n_estimators_{nEstim}_batchSize{nCandApply}_log.pdf')
    plt.close('all')

    # Score residuals vs XGBoost score
    for iBins, nBins in enumerate(nBinsList):
        resVsXGBoostFig = plt.figure(figsize=(8, 8))
        plt.hist2d(yPredTest['XGBoost'], yPredTest[f'BBDT{nBins}'][:, 1]-yPredTest['XGBoost'], cmap='OrRd',
                   range=np.array([(0., 1.), (-1., 1.)]), bins=(1000, 1000), norm=LogNorm(vmin=1.e-7))
        plt.ylabel(f'BDT output BBDT nbins = {nBins} - XGBoost', size=15)
        plt.xlabel(f'BDT output XGBoost', size=15)
        resVsXGBoostFig.savefig(f'plots/ResidualsBBDT{nBins}_vs_XGBoost_n_estimators_{nEstim}_batchSize{nCandApply}_lin.pdf')

    resVsXGBoostFig = plt.figure(figsize=(8, 8))
    plt.hist2d(yPredTest['XGBoost'], yPredTest['Treelite']-yPredTest['XGBoost'], cmap='OrRd',
               range=np.array([(0., 1.), (-1., 1.)]), bins=(1000, 1000), norm=LogNorm(vmin=1.e-7))
    plt.ylabel(f'BDT output Treelite - XGBoost', size=15)
    plt.xlabel(f'BDT output XGBoost', size=15)
    resVsXGBoostFig.savefig(f'plots/ResidualsTreelite_vs_XGBoost_n_estimators_{nEstim}_batchSize{nCandApply}_lin.pdf')
    plt.close('all')

dictTime = dict()
dictTime['n_estimators'] = nEstimList
dictTime['XGBoost'] = timeAppl['XGBoost']
dictTime['Treelite'] = timeAppl['Treelite']
for iBins, nBins in enumerate(nBinsList):
    dictTime[f'BBDT{nBins}'] = timeAppl[f'BBDT{nBins}']
dfTime = pd.DataFrame(dictTime)
dfTime.to_parquet(f'outputs/timetests/timeBench_XGBoost_v{xgb.__version__}_batchSize{nCandApply}'
                  f'_features_{"-".join(featuresForTrain)}.parquet.gzip',
                  compression='gzip')
