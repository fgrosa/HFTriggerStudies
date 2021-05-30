import uproot
from ROOT import TH1F, TH2F, TCanvas, kRainBow
from StyleFormatter import SetGlobalStyle, SetObjectStyle

SetGlobalStyle(padbottommargin=0.14, padleftmargin=0.15, palette=kRainBow, titleoffsety=1.5)

tree = uproot.open('AnalysisResults_0000.root:PWGHF_D2H_CharmTrigger_GenPurpose/fRecoTree')
dfBplus = tree.arrays(filter_name=['Beauty3Prong/Beauty3Prong.fInvMassBplustoD0pi',
                                   'Beauty3Prong/Beauty3Prong.fInvMassNoPropBplustoD0pi',
                                   'Beauty3Prong/Beauty3Prong.fDecayLength'], library='pd')
dfBzero = tree.arrays(filter_name=['Beauty4Prong/Beauty4Prong.fInvMassB0toDminuspi',
                                   'Beauty4Prong/Beauty4Prong.fInvMassNoPropB0toDminuspi',
                                   'Beauty4Prong/Beauty4Prong.fDecayLength'], library='pd')

dfBplus['delta'] = dfBplus['Beauty3Prong.fInvMassNoPropBplustoD0pi'] - dfBplus['Beauty3Prong.fInvMassBplustoD0pi']
dfBzero['delta'] = dfBzero['Beauty4Prong.fInvMassNoPropB0toDminuspi'] - dfBzero['Beauty4Prong.fInvMassB0toDminuspi']

hDeltaMassBplus = TH1F('hDeltaMassBplus', ';#it{M}(#bar{D}^{0}#pi^{+}, no vtx)-#it{M}(#bar{D}^{0}#pi^{+}, vtx) (MeV/#it{c}^{2});entries',1000, -1., 1.)
SetObjectStyle(hDeltaMassBplus, fillstyle=0, markersize=0.5)
hDeltaMassBplusVsDecL = TH2F('hDeltaMassBplusVsDecL',
                             ';decay length (cm);#it{M}(#bar{D}^{0}#pi^{+}, no vtx)-#it{M}(#bar{D}^{0}#pi^{+}, vtx) (MeV/#it{c}^{2});',
                             100, 0., 1., 1000, -1., 1.)

hDeltaMassBzero = TH1F('hDeltaMassBzero', ';#it{M}(D^{-}#pi^{+}, no vtx)-#it{M}(D^{-}#pi^{+}, vtx) (MeV/#it{c}^{2});entries',1000, -1., 1.)
SetObjectStyle(hDeltaMassBzero, fillstyle=0, markersize=0.5)
hDeltaMassBzeroVsDecL = TH2F('hDeltaMassBzeroVsDecL',
                             ';decay length (cm);#it{M}(D^{-}#pi^{+}, no vtx)-#it{M}(D^{-}#pi^{+}, vtx) (MeV/#it{c}^{2});',
                             100, 0., 1., 1000, -1., 1.)

for delta, decL in zip(dfBplus['delta'].to_numpy(), dfBplus['Beauty3Prong.fDecayLength'].to_numpy()):
    hDeltaMassBplus.Fill(delta*1000)
    hDeltaMassBplusVsDecL.Fill(decL, delta*1000)

for delta, decL, mass in zip(dfBzero['delta'].to_numpy(), dfBzero['Beauty4Prong.fDecayLength'].to_numpy(),
                             dfBzero['Beauty4Prong.fInvMassB0toDminuspi'].to_numpy()):
    if mass > 1000 or mass < 1:
        continue
    hDeltaMassBzero.Fill(delta*1000)
    hDeltaMassBzeroVsDecL.Fill(decL, delta*1000)

cDeltaMassBplus = TCanvas('cDeltaMassBplus', '', 1000, 500)
cDeltaMassBplus.Divide(2, 1)
cDeltaMassBplus.cd(1).SetLogy()
hDeltaMassBplus.Draw('e')
cDeltaMassBplus.cd(2).SetLogz()
hDeltaMassBplusVsDecL.Draw('colz')
cDeltaMassBplus.Modified()
cDeltaMassBplus.Update()

cDeltaMassBzero = TCanvas('cDeltaMassBzero', '', 1000, 500)
cDeltaMassBzero.Divide(2, 1)
cDeltaMassBzero.cd(1).SetLogy()
hDeltaMassBzero.Draw('e')
cDeltaMassBzero.cd(2).SetLogz()
hDeltaMassBzeroVsDecL.Draw('colz')
cDeltaMassBzero.Modified()
cDeltaMassBzero.Update()

cDeltaMassBplus.SaveAs('Bplus_mass_w_wo_vtx_propagation.pdf')
cDeltaMassBzero.SaveAs('Bzero_mass_w_wo_vtx_propagation.pdf')

input()