import sys
import os
import argparse
import uproot
import pandas as pd
import yaml
sys.path.append('..')
from pyutils.DfUtils import FilterBitDf

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('configFileName', metavar='text', default='config_skim_tree.yml')
args = parser.parse_args()

recoSelBits = {'Dzero': [0, 1], 'Dplus': [2], 'Ds': [4, 5], 'Lc': [6, 7],
               'Bplus': [9], 'Bzero': [10], 'Bs': [11], 'Lb': [12]}
decays = {'Dzero': [2, 1], 'Dplus': [3], 'Ds': [4, 5], 'Lc': [9, 10],
          'Bplus': [13], 'Bzero': [14], 'Bs': [15], 'Lb': [16]}
massNames = {'Dzero': ['fInvMassD0', 'fInvMassD0bar'],
             'Dplus': ['fInvMassDplus'],
             'Ds': ['fInvMassDstoKKpi', 'fInvMassDstopiKK'],
             'Lc': ['fInvMassLctopKpi', 'fInvMassLctopiKp'],
             'Bplus': ['fInvMassBplustoD0pi'],
             'Bzero': ['fInvMassB0toDminuspi'],
             'Bs': ['fInvMassBstoDsminuspi'],
             'Lb': ['fInvMassLbtoLcpluspi']}
fidAccBit = {'Dzero': 22, 'Dplus': 23, 'Ds': 25, 'Lc': 26,
             'Bplus': 27, 'Bzero': 28, 'Bs': 29, 'Lb': 30}

with open(args.configFileName, 'r') as ymlConfigFile:
    cfg = yaml.load(ymlConfigFile, yaml.FullLoader)

# reco candidates
treeReco = uproot.open(f'{cfg["infile"]["name"]}:{cfg["infile"]["dir"]}/fRecoTree')

if cfg['channels']['Dzero']['enable']:
    vars = [var for var in treeReco.keys() if ('Charm2Prong' in var) or var in ['Ntracklets', 'zVtxReco']]
    dfRecoDzero = treeReco.arrays(filter_name=vars, library='pd')
    for iVar, var in enumerate(vars):
        vars[iVar] = var.replace('Charm2Prong/Charm2Prong.', '')
    if 'Charm2Prong' in vars:
        vars.remove('Charm2Prong')
    dfRecoDzero.columns = vars
    dfRecoBkg = FilterBitDf(dfRecoDzero, 'fCandType', [1])
    dfRecoPrompt = FilterBitDf(dfRecoDzero, 'fCandType', [2])
    dfRecoFD = FilterBitDf(dfRecoDzero, 'fCandType', [3])
    for iDec, (recoSelBit, decay, massName) in enumerate(zip(recoSelBits['Dzero'], decays['Dzero'], massNames['Dzero'])):
        dfRecoPromptSel = dfRecoPrompt.query(f'fDecay == {decay}')
        dfRecoFDSel = dfRecoFD.query(f'fDecay == {decay}')
        dfRecoBkgSel = FilterBitDf(dfRecoBkg, 'fSelBit', [recoSelBit, fidAccBit['Dzero']], 'and')
        dfRecoPromptSel = FilterBitDf(dfRecoPromptSel, 'fSelBit', [recoSelBit, fidAccBit['Dzero']], 'and')
        dfRecoFDSel = FilterBitDf(dfRecoFDSel, 'fSelBit', [recoSelBit, fidAccBit['Dzero']], 'and')
        dfRecoBkgSel = dfRecoBkgSel.rename(columns = {massName: 'fInvMass'})
        dfRecoPromptSel = dfRecoPromptSel.rename(columns = {massName: 'fInvMass'})
        dfRecoFDSel = dfRecoFDSel.rename(columns = {massName: 'fInvMass'})
        if cfg['channels']['Dzero']['vars'] is None or len(cfg['channels']['Dzero']['vars']) == 0:
            varsToSave = vars.copy()
            varsToSave.remove('fDecay')
            varsToSave.remove('fSelBit')
            varsToSave.remove('fCandType')
            for massNameToRemove in massNames['Dzero']:
                varsToSave.remove(massNameToRemove)                
        else:
            varsToSave = cfg['channels']['Dzero']['vars']
            if 'fInvMass' not in varsToSave:
                print('WARNING: invariant mass not included in variables to save, adding it')
        varsToSave.append('fInvMass')
        dfRecoBkgSel = dfRecoBkgSel[varsToSave]
        dfRecoPromptSel = dfRecoPromptSel[varsToSave]
        dfRecoFDSel = dfRecoFDSel[varsToSave]
        dfRecoBkgSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_bkg_D0.parquet_{iDec}.gzip'),
                                compression='gzip')
        dfRecoPromptSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_prompt_D0.parquet_{iDec}.gzip'),
                                   compression='gzip')
        dfRecoFDSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_FD_D0.parquet_{iDec}.gzip'),
                               compression='gzip')

if cfg['channels']['Dplus']['enable'] or cfg['channels']['Ds']['enable'] or cfg['channels']['Lc']['enable']:
    vars = [var for var in treeReco.keys() if ('Charm3Prong' in var) or var in ['Ntracklets', 'zVtxReco']]
    dfReco3Prong = treeReco.arrays(filter_name=vars, library='pd')
    for iVar, var in enumerate(vars):
        vars[iVar] = var.replace('Charm3Prong/Charm3Prong.', '')
    if 'Charm3Prong' in vars:
        vars.remove('Charm3Prong')
    dfReco3Prong.columns = vars
    dfRecoBkg = FilterBitDf(dfReco3Prong, 'fCandType', [1])
    dfRecoPrompt = FilterBitDf(dfReco3Prong, 'fCandType', [2])
    dfRecoFD = FilterBitDf(dfReco3Prong, 'fCandType', [3])
    for species in ['Dplus', 'Ds', 'Lc']:
        for recoSelBit, decay, massName in zip(recoSelBits[species], decays[species], massNames[species]):
            dfRecoPromptSel = dfRecoPrompt.query(f'fDecay == {decay}')
            dfRecoFDSel = dfRecoFD.query(f'fDecay == {decay}')
            dfRecoBkgSel = FilterBitDf(dfRecoBkg, 'fSelBit', [recoSelBit, fidAccBit[species]], 'and')
            dfRecoPromptSel = FilterBitDf(dfRecoPromptSel, 'fSelBit', [recoSelBit, fidAccBit[species]], 'and')
            dfRecoFDSel = FilterBitDf(dfRecoFDSel, 'fSelBit', [recoSelBit, fidAccBit[species]], 'and')
            dfRecoBkgSel['fInvMass'] = dfRecoBkgSel[massName]
            dfRecoPromptSel['fInvMass'] = dfRecoPromptSel[massName]
            dfRecoFDSel['fInvMass'] = dfRecoFDSel[massName]
            if cfg['channels'][species]['vars'] is None or len(cfg['channels'][species]['vars']) == 0:
                varsToSave = vars.copy()
                varsToSave.remove('fDecay')
                varsToSave.remove('fSelBit')
                varsToSave.remove('fCandType')
            else:
                varsToSave = cfg['channels'][species]['vars']
                if 'fInvMass' not in varsToSave:
                    print('WARNING: invariant mass not included in variables to save, adding it')
                    varsToSave.append('fInvMass')
            dfRecoBkgSel = dfRecoBkgSel[varsToSave]
            dfRecoPromptSel = dfRecoPromptSel[varsToSave]
            dfRecoFDSel = dfRecoFDSel[varsToSave]
            dfRecoBkgSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_bkg_{species}.parquet.gzip'),
                                    compression='gzip')
            dfRecoPromptSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_prompt_{species}.parquet.gzip'),
                                       compression='gzip')
            dfRecoFDSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_FD_{species}.parquet.gzip'),
                                   compression='gzip')
            
if cfg['channels']['Bplus']['enable']:
    vars = [var for var in treeReco.keys() if ('Beauty3Prong' in var) or var in ['Ntracklets', 'zVtxReco']]
    dfRecoDzero = treeReco.arrays(filter_name=vars, library='pd')
    for iVar, var in enumerate(vars):
        vars[iVar] = var.replace('Beauty3Prong/Beauty3Prong.', '')
    if 'Beauty3Prong' in vars:
        vars.remove('Beauty3Prong')
    dfRecoDzero.columns = vars
    dfRecoBkg = FilterBitDf(dfRecoDzero, 'fCandType', [1])
    dfRecoPrompt = FilterBitDf(dfRecoDzero, 'fCandType', [2])
    for recoSelBit, decay, massName in zip(recoSelBits['Bplus'], decays['Bplus'], massNames['Bplus']):
        dfRecoPromptSel = dfRecoPrompt.query(f'fDecay == {decay}')
        dfRecoBkgSel = FilterBitDf(dfRecoBkg, 'fSelBit', [recoSelBit, fidAccBit['Bplus']], 'and')
        dfRecoPromptSel = FilterBitDf(dfRecoPromptSel, 'fSelBit', [recoSelBit, fidAccBit['Bplus']], 'and')
        dfRecoBkgSel['fInvMass'] = dfRecoBkgSel[massName]
        dfRecoPromptSel['fInvMass'] = dfRecoPromptSel[massName]
        if cfg['channels']['Bplus']['vars'] is None or len(cfg['channels']['Bplus']['vars']) == 0:
            varsToSave = vars.copy()
            varsToSave.remove('fDecay')
            varsToSave.remove('fSelBit')
            varsToSave.remove('fCandType')
        else:
            varsToSave = cfg['channels']['Bplus']['vars']
            if 'fInvMass' not in varsToSave:
                print('WARNING: invariant mass not included in variables to save, adding it')
                varsToSave.append('fInvMass')
        dfRecoBkgSel = dfRecoBkgSel[varsToSave]
        dfRecoPromptSel = dfRecoPromptSel[varsToSave]
        dfRecoBkgSel.to_parquet(os.path.join(cfg['outfiles']['dir'], 'Reco_bkg_Bplus.parquet.gzip'),
                                compression='gzip')
        dfRecoPromptSel.to_parquet(os.path.join(cfg['outfiles']['dir'], 'Reco_Bplus.parquet.gzip'),
                                   compression='gzip')

if cfg['channels']['Bzero']['enable'] or cfg['channels']['Bs']['enable'] or cfg['channels']['Lb']['enable']:
    vars = [var for var in treeReco.keys() if ('Beauty4Prong' in var) or var in ['Ntracklets', 'zVtxReco']]
    dfReco3Prong = treeReco.arrays(filter_name=vars, library='pd')
    for iVar, var in enumerate(vars):
        vars[iVar] = var.replace('Beauty4Prong/Beauty4Prong.', '')
    if 'Beauty4Prong' in vars:
        vars.remove('Beauty4Prong')
    dfReco3Prong.columns = vars
    dfRecoBkg = FilterBitDf(dfReco3Prong, 'fCandType', [1])
    dfRecoPrompt = FilterBitDf(dfReco3Prong, 'fCandType', [2])
    for species in ['Bzero', 'Bs', 'Lb']:
        for recoSelBit, decay, massName in zip(recoSelBits[species], decays[species], massNames[species]):
            dfRecoPromptSel = dfRecoPrompt.query(f'fDecay == {decay}')
            dfRecoBkgSel = FilterBitDf(dfRecoBkg, 'fSelBit', [recoSelBit, fidAccBit[species]], 'and')
            dfRecoPromptSel = FilterBitDf(dfRecoPromptSel, 'fSelBit', [recoSelBit, fidAccBit[species]], 'and')
            dfRecoBkgSel['fInvMass'] = dfRecoBkgSel[massName]
            dfRecoPromptSel['fInvMass'] = dfRecoPromptSel[massName]
            if cfg['channels'][species]['vars'] is None or len(cfg['channels'][species]['vars']) == 0:
                varsToSave = vars.copy()
                varsToSave.remove('fDecay')
                varsToSave.remove('fSelBit')
                varsToSave.remove('fCandType')
            else:
                varsToSave = cfg['channels'][species]['vars']
                if 'fInvMass' not in varsToSave:
                    print('WARNING: invariant mass not included in variables to save, adding it')
                    varsToSave.append('fInvMass')
            dfRecoBkgSel = dfRecoBkgSel[varsToSave]
            dfRecoPromptSel = dfRecoPromptSel[varsToSave]
            dfRecoBkgSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_bkg_{species}.parquet.gzip'),
                                    compression='gzip')
            dfRecoPromptSel.to_parquet(os.path.join(cfg['outfiles']['dir'], f'Reco_{species}.parquet.gzip'),
                                       compression='gzip')

# gen candidates
#for iFile, file in enumerate(fileNameList):
#    treeGen = uproot.open(f'{file}:{cfg["infile"]["dir"]}/fRecoTree')

# treeGen = uproot.open(f'{cfg["infile"]["name"]}:{cfg["infile"]["dir"]}/fGenTree')

# if cfg['channels']['Dzero']['gen'] == 'tree':
    
