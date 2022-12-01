'''
Script for the comparison of trigger class and rejection factor with or without BDT selection
run: python compare_bdt_variation.py cfgFileName.yml
'''

import sys
from os.path import join
import argparse
import numpy as np
from itertools import zip_longest
import yaml
from ROOT import TCanvas, TFile, TLegend, TLine, gPad  # pylint: disable=import-error,no-name-in-module

sys.path.append('../..')
from pyutils.StyleFormatter import SetGlobalStyle, SetObjectStyle, GetROOTColor, GetROOTMarker  #pylint: disable=wrong-import-position,import-error

# load inputs
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('cfgFileName',
                    metavar='text',
                    default='config_comparison.yml')
args = parser.parse_args()

with open(args.cfgFileName, 'r') as ymlCfgFile:
    inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)

inDirName = inputCfg['inputs']['dirname']
inFileNames = inputCfg['inputs']['filenames']

particle_c = inputCfg['inputs']['charmhadron']
particle_b = inputCfg['inputs']['beautyhadron']

outFileName = inputCfg['output']['filename']

colors = inputCfg['options']['colors']
markers = inputCfg['options']['markers']
markersize = inputCfg['options']['markersize']
linewidth = inputCfg['options']['linewidth']
fillstyles = inputCfg['options']['fillstyle']

wCanv = inputCfg['options']['canvas']['width']
hCanv = inputCfg['options']['canvas']['heigth']
xLimitslow_c = inputCfg['options']['histos']['xLimitslow_c']
xLimitshigh_c = inputCfg['options']['histos']['xLimitshigh_c']
xLimitslow_b = inputCfg['options']['histos']['xLimitslow_b']
xLimitshigh_b = inputCfg['options']['histos']['xLimitshigh_b']

xTitle_c = inputCfg['options']['histos']['xaxistitle_c']
xTitle_b = inputCfg['options']['histos']['xaxistitle_b']
xTitle_h = inputCfg['options']['histos']['xaxistitle_hightpt']
xTitle_k = inputCfg['options']['histos']['xaxistitle_kstar']

yTitle = inputCfg['options']['histos']['yaxistitle']
ptlow = inputCfg['options']['histos']['ptlow']
pthigh = inputCfg['options']['histos']['pthigh']
kstarlow = inputCfg['options']['histos']['kstarlow']
kstarhigh = inputCfg['options']['histos']['kstarhigh']

xLegLimits = inputCfg['options']['legend']['xlimits']
yLegLimits = inputCfg['options']['legend']['ylimits']
legHeader = inputCfg['options']['legend']['header']
legNames = inputCfg['options']['legend']['titles']
legOpt = inputCfg['options']['legend']['options']
legTextSize = inputCfg['options']['legend']['textsize']
ncolumns = inputCfg['options']['legend']['ncolumns']

# set global style
SetGlobalStyle(padbottommargin=0.14,
               padtopmargin=0.08,
               padleftmargin=0.15,
               padrightmargin=0.1,
               titleoffsety=1.45,
               titleoffsetx=1.1,
               titlesize=0.05,
               labelsize=0.05,
               maxdigits=2,
               opttitle=1)

CharmMass = [[], []]
BeautyMass = [[], []]
HighPt = [[], []]
ProtonKstar = [[], []]
ReJFactor = []
Nevents = []

for iFile, (inFileName) in enumerate(inFileNames):
    if inDirName:
        inFileName = join(inDirName, inFileName)
    inFile = TFile.Open(inFileName)
    if inFile == None:
        print(f"ERROR: cannot open {inFileName}. Check your config. Exit!")
        sys.exit()
    Nevents.append(
        inFile.Get('hf-filter/registry/fProcessedEvents').GetBinContent(1))
    ReJFactor.append(inFile.Get('hf-filter/registry/fProcessedEvents'))
    print(iFile, Nevents[iFile])
    ReJFactor[iFile].Scale(1 / Nevents[iFile])
    ReJFactor[iFile].SetNameTitle(f'hRej{iFile}', f'hRej{iFile}')
    ReJFactor[-1].SetDirectory(0)

    for i, (parc, parb) in enumerate(zip(particle_c, particle_b)):

        CharmMass[iFile].append(
            inFile.Get(f'hf-filter/registry/fMassVsPt{parc}').ProjectionY())
        CharmMass[iFile][i].Scale(1 / Nevents[iFile])
        CharmMass[iFile][i].SetNameTitle(f'{parc}{iFile}', f'{parc}{iFile}')
        CharmMass[iFile][-1].SetDirectory(0)

        BeautyMass[iFile].append(
            inFile.Get(f'hf-filter/registry/fMassVsPt{parb}').ProjectionY())
        BeautyMass[iFile][i].Scale(1 / Nevents[iFile])
        BeautyMass[iFile][i].SetNameTitle(f'{parb}{iFile}', f'{parb}{iFile}')
        BeautyMass[iFile][-1].SetDirectory(0)

        if i < 4:
            HighPt[iFile].append(
                inFile.Get(f'hf-filter/registry/f{parc}HighPt'))
            HighPt[iFile][i].Scale(1 / Nevents[iFile])
            HighPt[iFile][i].SetNameTitle(f'Highpt{parc}{iFile}',
                                          f'Highpt{parc}{iFile}')
            HighPt[iFile][-1].SetDirectory(0)

            ProtonKstar[iFile].append(
                inFile.Get(f'hf-filter/registry/f{parc}ProtonKstarDistr'))
            ProtonKstar[iFile][i].Scale(1 / Nevents[iFile])
            ProtonKstar[iFile][i].SetNameTitle(f'hkstar{parb}{iFile}',
                                               f'hkstar{parb}{iFile}')
            ProtonKstar[iFile][-1].SetDirectory(0)

legRej = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
legRej.SetFillStyle(0)
legRej.SetTextSize(legTextSize)
legRej.SetNColumns(ncolumns)
if legHeader is not None:
    legRej.SetHeader(legHeader, 'C')

legCmass = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
legCmass.SetFillStyle(0)
legCmass.SetTextSize(legTextSize)
legCmass.SetNColumns(ncolumns)
if legHeader is not None:
    legCmass.SetHeader(legHeader, 'C')

legBmass = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
legBmass.SetFillStyle(0)
legBmass.SetTextSize(legTextSize)
legBmass.SetNColumns(ncolumns)
if legHeader is not None:
    legBmass.SetHeader(legHeader, 'C')

legHighpt = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
legHighpt.SetFillStyle(0)
legHighpt.SetTextSize(legTextSize)
legHighpt.SetNColumns(ncolumns)
if legHeader is not None:
    legHighpt.SetHeader(legHeader, 'C')

legKstar = TLegend(xLegLimits[0], yLegLimits[0], xLegLimits[1], yLegLimits[1])
legKstar.SetFillStyle(0)
legKstar.SetTextSize(legTextSize)
legKstar.SetNColumns(ncolumns)
if legHeader is not None:
    legKstar.SetHeader(legHeader, 'C')

cRejFac = TCanvas('cRejFac', '', 1000, 800)
cCharmMass = TCanvas('cCharmMass', '', wCanv, hCanv)
cBeautyMass = TCanvas('cBeautyMass', '', wCanv, hCanv)
cHighPt = TCanvas('cHighPt', '', wCanv, hCanv)
cKstar = TCanvas('cKstar', '', wCanv, hCanv)

cCharmMass.Divide(3, 2)
cBeautyMass.Divide(3, 2)
cHighPt.Divide(3, 2)
cKstar.Divide(3, 2)

line = TLine(-0.5, 1, 10.5, 1)
line.SetLineColor(1)
line.SetLineWidth(2)
line.SetLineStyle(2)

for i, (hrejfactor, hcharmmass, hbeautymass, hhighpt, hkstar, color, marker,
        fillstyle) in enumerate(
            zip(ReJFactor, CharmMass, BeautyMass, HighPt, ProtonKstar, colors,
                markers, fillstyles)):

    cRejFac.cd()
    hrejfactor.SetBinContent(2, 1 - hrejfactor.GetBinContent(2))
    hrejfactor.GetXaxis().SetBinLabel(2, "accpected")
    hrejfactor.SetTitle("Rejection fractor;;Rejection fractor")
    gPad.SetLogy()
    gPad.SetGridx()
    gPad.SetGridy()
    print(hrejfactor.GetBinContent(2))
    legRej.AddEntry(hrejfactor, legNames[i], legOpt[i])
    hrejfactor.Draw('same')
    legRej.Draw('same')
    line.Draw()
    hrejfactor.GetYaxis().SetRangeUser(hrejfactor.GetMinimum() * 0.002,
                                       hrejfactor.GetMaximum() * 50)
    SetObjectStyle(hrejfactor,
                   color=GetROOTColor(color),
                   markerstyle=GetROOTMarker(marker),
                   markersize=markersize,
                   linewidth=linewidth,
                   fillstyle=fillstyle)

    for p, (par_c, par_b, xl_c, xh_c, xl_b, xh_b, xT_c, xT_b, yT, pl, ph, kl,
            kh) in enumerate(
                zip_longest(particle_c, particle_b, xLimitslow_c,
                            xLimitshigh_c, xLimitslow_b, xLimitshigh_b,
                            xTitle_c, xTitle_b, yTitle, ptlow, pthigh,
                            kstarlow, kstarhigh)):
        if i == 0:
            cCharmMass.cd(p + 1).DrawFrame(xl_c,
                                           hcharmmass[p].GetMaximum() * 10e-5,
                                           xh_c,
                                           hcharmmass[p].GetMaximum() * 20,
                                           f'{par_c};{xT_c};{yT}')
            hcharmmass[p].Draw('same e')
            gPad.SetLogy()
            cBeautyMass.cd(p + 1).DrawFrame(
                xl_b, hbeautymass[p].GetMaximum() * 10e-5, xh_b,
                hbeautymass[p].GetMaximum() * 30, f'{par_b};{xT_b};{yT}')
            hbeautymass[p].Draw('same e')
            gPad.SetLogy()

            if p < 4:
                cHighPt.cd(p + 1).DrawFrame(pl,
                                            hhighpt[p].GetMaximum() * 10e-5,
                                            ph, hhighpt[p].GetMaximum() * 10,
                                            f'{par_c};{xTitle_h};{yT}')
                hhighpt[p].Draw('same e')
                gPad.SetLogy()

                cKstar.cd(p + 1).DrawFrame(kl, hkstar[p].GetMaximum() * 10e-5,
                                           kh, hkstar[p].GetMaximum() * 10,
                                           f'{par_c};{xTitle_k};{yT}')
                hkstar[p].Draw('same e')
                gPad.SetLogy()

        else:
            cCharmMass.cd(p + 1)
            hcharmmass[p].Draw('same e')
            cBeautyMass.cd(p + 1)
            hbeautymass[p].Draw('same e')
            if p < 4:
                cHighPt.cd(p + 1)
                hhighpt[p].Draw('same e')
                cKstar.cd(p + 1)
                hkstar[p].Draw('same e')

        if p == 0:
            cCharmMass.cd(p + 1)
            legCmass.AddEntry(hcharmmass[p], legNames[i], legOpt[i])
            legCmass.Draw()
            cBeautyMass.cd(p + 1)
            legBmass.AddEntry(hbeautymass[p], legNames[i], legOpt[i])
            legBmass.Draw()
            cHighPt.cd(p + 1)
            legHighpt.AddEntry(hhighpt[p], legNames[i], legOpt[i])
            legHighpt.Draw()
            cKstar.cd(p + 1)
            legKstar.AddEntry(hkstar[p], legNames[i], legOpt[i])
            legKstar.Draw()

        SetObjectStyle(hcharmmass[p],
                       color=GetROOTColor(color),
                       markerstyle=GetROOTMarker(marker),
                       markersize=markersize,
                       linewidth=linewidth,
                       fillstyle=fillstyle)
        SetObjectStyle(hbeautymass[p],
                       color=GetROOTColor(color),
                       markerstyle=GetROOTMarker(marker),
                       markersize=markersize,
                       linewidth=linewidth,
                       fillstyle=fillstyle)
        if p < 4:
            SetObjectStyle(hhighpt[p],
                           color=GetROOTColor(color),
                           markerstyle=GetROOTMarker(marker),
                           markersize=markersize,
                           linewidth=linewidth,
                           fillstyle=fillstyle)
            SetObjectStyle(hkstar[p],
                           color=GetROOTColor(color),
                           markerstyle=GetROOTMarker(marker),
                           markersize=markersize,
                           linewidth=linewidth,
                           fillstyle=fillstyle)

outFile = TFile(f'{outFileName}.root', 'recreate')
cRejFac.Write()
cCharmMass.Write()
cBeautyMass.Write()
cHighPt.Write()
cKstar.Write()
cRejFac.SaveAs("RecFractor.pdf")
cCharmMass.SaveAs("CharmInvMass.pdf")
cBeautyMass.SaveAs("BeautyInvMass.pdf")
cHighPt.SaveAs("HighPt.pdf")
cKstar.SaveAs("ProtonKstarDis.pdf")

input("Press enter to exit")
