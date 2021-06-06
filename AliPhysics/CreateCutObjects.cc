#if !defined (__CINT__) || defined (__CLING__)

#include <TFile.h>
#include "AliESDtrackCuts.h"
#include "AliRDHFCutsD0toKpi.h"
#include "AliRDHFCutsDplustoKpipi.h"
#include "AliRDHFCutsDstoKKpi.h"
#include "AliRDHFCutsLctopKpi.h"
#include "AliAODPidHF.h"
#include "AliPID.h"

#endif

void SetupCombinedPID(AliRDHFCutsLctopKpi *cutsObj,double threshold) {

  cutsObj->GetPidHF()->SetCombDetectors(AliAODPidHF::kTPCTOF);
  for (int ispecies=0;ispecies<AliPID::kSPECIES;++ispecies)
    cutsObj->SetPIDThreshold(static_cast<AliPID::EParticleType>(ispecies),threshold);
  cutsObj->GetPidHF()->SetUseCombined(true);
  cutsObj->GetPidHF()->SetUpCombinedPID();
  return;
}

void CreateCutObjects(TString outfilename = "D0DplusDsLcPreselectionCuts.root")
{
    AliESDtrackCuts* esdTrackCuts = new AliESDtrackCuts();
    esdTrackCuts->SetRequireSigmaToVertex(false);
    esdTrackCuts->SetRequireTPCRefit(true);
    esdTrackCuts->SetRequireITSRefit(true);
    esdTrackCuts->SetClusterRequirementITS(AliESDtrackCuts::kSPD, AliESDtrackCuts::kAny);
    esdTrackCuts->SetEtaRange(-0.8, 0.8);
    esdTrackCuts->SetMinDCAToVertexXY(0.);
    esdTrackCuts->SetPtRange(0.3, 1.e10);
    esdTrackCuts->SetMaxDCAToVertexXY(1.e6);

    const int nptbins = 2;
    float* ptbins;
    ptbins = new float[nptbins+1];
    ptbins[0]=0.;
    ptbins[1]=8.;
    ptbins[2]=100.;

    //D0_____________________________________________________________________________________
    const int nvarsD0 = 11;
    AliRDHFCutsD0toKpi* cutsD0toKpi=new AliRDHFCutsD0toKpi();
    cutsD0toKpi->SetName("D0toKpiCuts");
    cutsD0toKpi->AddTrackCuts(esdTrackCuts);
    cutsD0toKpi->SetPtBins(nptbins+1, ptbins);
    cutsD0toKpi->SetGlobalIndex(nvarsD0, nptbins);

    //m    dca      cost*  ptk ptpi  d0k          d0pi       d0d0          cosp  cosxy normdxy
    float cutsMatrixD0toKpiStand[nptbins][nvarsD0]=  {
        {0.300, 300.*1E-4, 0.9, 0.3, 0.3, 1000.*1E-4, 1000.*1E-4, 100000. *1E-8, 0.7,  0., 0.},
        {0.300, 300.*1E-4, 1.0, 0.3, 0.3, 1000.*1E-4, 1000.*1E-4, 100000. *1E-8, 0.7, 0., 0.}};

    float **cutsMatrixTransposeStand = new float*[nvarsD0];
    for(int iv=0;iv<nvarsD0;iv++)
        cutsMatrixTransposeStand[iv] = new float[nptbins];

    for (int ibin=0;ibin<nptbins;ibin++)
    {
        for (int ivar = 0; ivar<nvarsD0; ivar++)
        {
            cutsMatrixTransposeStand[ivar][ibin]=cutsMatrixD0toKpiStand[ibin][ivar];
        }
    }

    cutsD0toKpi->SetCuts(nvarsD0, nptbins, cutsMatrixTransposeStand);
    cutsD0toKpi->SetUseSpecialCuts(false);
    for(int iv = 0; iv < nvarsD0; iv++)
        delete [] cutsMatrixTransposeStand[iv];
    delete [] cutsMatrixTransposeStand;
    cutsMatrixTransposeStand = NULL;

    //pid settings
    AliAODPidHF* pidObj = new AliAODPidHF();
    int mode = 1;
    const int nlims = 2;
    double plims[nlims] = {0.6, 0.8}; //TPC limits in momentum [GeV/c]
    bool compat = true; //effective only for this mode
    bool asym = true;
    double sigmas[5] = {2., 1., 0., 3., 0.}; //to be checked and to be modified with new implementation of setters by Rossella
    pidObj->SetAsym(asym);// if you want to use the asymmetric bands in TPC
    pidObj->SetMatch(mode);
    pidObj->SetPLimit(plims,nlims);
    pidObj->SetSigma(sigmas);
    pidObj->SetCompat(compat);
    pidObj->SetTPC(true);
    pidObj->SetTOF(true);
    pidObj->SetPCompatTOF(1.5);
    pidObj->SetSigmaForTPCCompat(3.);
    pidObj->SetSigmaForTOFCompat(3.);
    pidObj->SetOldPid(false);
    cutsD0toKpi->SetPidHF(pidObj);
    cutsD0toKpi->SetUsePID(true);
    cutsD0toKpi->SetMinPtCandidate(0.);
    cutsD0toKpi->SetMaxPtCandidate(100.);

    cutsD0toKpi->PrintAll();

    //D+_____________________________________________________________________________________
    AliRDHFCutsDplustoKpipi* cutsDplustoKpipi = new AliRDHFCutsDplustoKpipi();
    cutsDplustoKpipi->SetName("DplustoKpipiCuts");
    cutsDplustoKpipi->AddTrackCuts(esdTrackCuts);

    const int nvarsDplus=14;
    float** anacutsvalDplus;
    anacutsvalDplus=new float*[nvarsDplus];
    for(int ic=0; ic<nvarsDplus; ic++)
        anacutsvalDplus[ic] = new float[nptbins];

    anacutsvalDplus[0][0]=0.2;    //minv
    anacutsvalDplus[1][0]=0.4;    //ptK
    anacutsvalDplus[2][0]=0.4;    //ptpi
    anacutsvalDplus[3][0]=0.;     //d0K
    anacutsvalDplus[4][0]=0.;     //d0pi
    anacutsvalDplus[5][0]=0.;     //dist12
    anacutsvalDplus[6][0]=0.030;  //sigvert
    anacutsvalDplus[7][0]=0.03;   //decay length
    anacutsvalDplus[8][0]=0.0;    //pM
    anacutsvalDplus[9][0]=0.75;   //cosp
    anacutsvalDplus[10][0]=0.0;   //sumd02
    anacutsvalDplus[11][0]=1.e10; //dca
    anacutsvalDplus[12][0]=0.;    //norm dec len xy
    anacutsvalDplus[13][0]=0.75;  //cospXY

    anacutsvalDplus[0][1]=0.2;    //minv
    anacutsvalDplus[1][1]=0.4;    //ptK
    anacutsvalDplus[2][1]=0.4;    //ptpi
    anacutsvalDplus[3][1]=0.;     //d0K
    anacutsvalDplus[4][1]=0.;     //d0pi
    anacutsvalDplus[5][1]=0.;     //dist12
    anacutsvalDplus[6][1]=0.060;  //sigvert
    anacutsvalDplus[7][1]=0.03;   //decay length
    anacutsvalDplus[8][1]=0.0;    //pM
    anacutsvalDplus[9][1]=0.75;   //cosp
    anacutsvalDplus[10][1]=0.0;   //sumd02
    anacutsvalDplus[11][1]=1.e10; //dca
    anacutsvalDplus[12][1]=0.;    //norm dec len xy
    anacutsvalDplus[13][1]=0.75;  //cospXY

    cutsDplustoKpipi->SetPtBins(nptbins+1,ptbins);
    cutsDplustoKpipi->SetCuts(nvarsDplus,nptbins,anacutsvalDplus);
    cutsDplustoKpipi->SetScaleNormDLxyBypOverPt(false);
    cutsDplustoKpipi->SetUseImpParProdCorrCut(false);
    cutsDplustoKpipi->SetUsePID(true);
    cutsDplustoKpipi->SetMinPtCandidate(0.);
    cutsDplustoKpipi->SetMaxPtCandidate(100.);

    cutsDplustoKpipi->PrintAll();

    //Ds+_____________________________________________________________________________________
    AliRDHFCutsDstoKKpi* cutsDstoKKpi = new AliRDHFCutsDstoKKpi();
    cutsDstoKKpi->SetName("DstoKKpiCuts");
    cutsDstoKKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsDs = 20;
    float** anacutsvalDs;
    anacutsvalDs=new float*[nvarsDs];
    for(int ic=0;ic<nvarsDs;ic++)
        anacutsvalDs[ic]=new float[nptbins];

    anacutsvalDs[0][0]=0.35; //0 inv. mass [GeV]
    anacutsvalDs[1][0]=0.3; //1 pTK [GeV/c]
    anacutsvalDs[2][0]=0.3; //2 pTPi [GeV/c]
    anacutsvalDs[3][0]=0.; //3 d0K [cm]
    anacutsvalDs[4][0]=0.; //4 d0pi [cm]
    anacutsvalDs[5][0]=0.; //5 dist12 [cm]
    anacutsvalDs[6][0]=0.03; //6 sigmavert [cm]
    anacutsvalDs[7][0]=0.02; //7 decLen [cm]",
    anacutsvalDs[8][0]=0.; //8 ptMax [GeV/c]
    anacutsvalDs[9][0]=0.7; //9 cosThetaPoint
    anacutsvalDs[10][0]=0.; //10 Sum d0^2 (cm^2)
    anacutsvalDs[11][0]=1000.; //11 dca [cm]
    anacutsvalDs[12][0]=0.020; //12 inv. mass (Mphi-MKK) [GeV]
    anacutsvalDs[13][0]=0.2; //13 inv. mass (MKo*-MKpi) [GeV]
    anacutsvalDs[14][0]=0.; //14 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[15][0]=1.; //15 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[16][0]=0.; //16 decLenXY [cm]
    anacutsvalDs[17][0]=0.; //17 NormdecLen
    anacutsvalDs[18][0]=0.; //18 NormdecLenXY
    anacutsvalDs[19][0]=0.7; //19 cosThetaPointXY

    anacutsvalDs[0][1]=0.35; //0 inv. mass [GeV]
    anacutsvalDs[1][1]=0.3; //1 pTK [GeV/c]
    anacutsvalDs[2][1]=0.3; //2 pTPi [GeV/c]
    anacutsvalDs[3][1]=0.; //3 d0K [cm]
    anacutsvalDs[4][1]=0.; //4 d0pi [cm]
    anacutsvalDs[5][1]=0.; //5 dist12 [cm]
    anacutsvalDs[6][1]=0.06; //6 sigmavert [cm]
    anacutsvalDs[7][1]=0.02; //7 decLen [cm]",
    anacutsvalDs[8][1]=0.; //8 ptMax [GeV/c]
    anacutsvalDs[9][1]=0.7; //9 cosThetaPoint
    anacutsvalDs[10][1]=0.; //10 Sum d0^2 (cm^2)
    anacutsvalDs[11][1]=1000.; //11 dca [cm]
    anacutsvalDs[12][1]=0.020; //12 inv. mass (Mphi-MKK) [GeV]
    anacutsvalDs[13][1]=0.2; //13 inv. mass (MKo*-MKpi) [GeV]
    anacutsvalDs[14][1]=0.; //14 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[15][1]=1.; //15 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[16][1]=0.; //16 decLenXY [cm]
    anacutsvalDs[17][1]=0.; //17 NormdecLen
    anacutsvalDs[18][1]=0.; //18 NormdecLenXY
    anacutsvalDs[19][1]=0.7; //19 cosThetaPointXY

    cutsDstoKKpi->SetPtBins(nptbins+1,ptbins);
    cutsDstoKKpi->SetCuts(nvarsDs,nptbins,anacutsvalDs);
    cutsDstoKKpi->SetUsePID(true);
    cutsDstoKKpi->SetPidOption(0); //0=kConservative,1=kStrong
    cutsDstoKKpi->SetMinPtCandidate(0.);
    cutsDstoKKpi->SetMaxPtCandidate(100.);

    cutsDstoKKpi->PrintAll();

    //Lc+_____________________________________________________________________________________
    AliRDHFCutsLctopKpi* cutsLctopKpi=new AliRDHFCutsLctopKpi();
    cutsLctopKpi->SetName("LctopKpiCuts");
    cutsLctopKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsLc = 13;
    float** anacutsvalLc;
    anacutsvalLc=new float*[nvarsDs];
    for(int ic=0;ic<nvarsLc;ic++)
        anacutsvalLc[ic]=new float[nptbins];

    anacutsvalLc[0][0] = 0.18;
    anacutsvalLc[1][0] = 0.3;
    anacutsvalLc[2][0] = 0.3;
    anacutsvalLc[3][0] = 0.;
    anacutsvalLc[4][0] = 0.;
    anacutsvalLc[5][0] = 0.;
    anacutsvalLc[6][0] = 0.04;
    anacutsvalLc[7][0] = 0.;
    anacutsvalLc[8][0] = 0.;
    anacutsvalLc[9][0] = -1.;
    anacutsvalLc[10][0] = 0.;
    anacutsvalLc[11][0] = 0.05;
    anacutsvalLc[12][0] = 0.3;

    anacutsvalLc[0][1] = 0.18;
    anacutsvalLc[1][1] = 0.3;
    anacutsvalLc[2][1] = 0.3;
    anacutsvalLc[3][1] = 0.;
    anacutsvalLc[4][1] = 0.;
    anacutsvalLc[5][1] = 0.;
    anacutsvalLc[6][1] = 0.06;
    anacutsvalLc[7][1] = 0.;
    anacutsvalLc[8][1] = 0.;
    anacutsvalLc[9][1] = -1.;
    anacutsvalLc[10][1] = 0.;
    anacutsvalLc[11][1] = 0.05;
    anacutsvalLc[12][1] = 0.3;

    cutsLctopKpi->SetUseTrackSelectionWithFilterBits(true);
    cutsLctopKpi->SetPtBins(nptbins+1,ptbins);
    cutsLctopKpi->SetCuts(nvarsLc,nptbins,anacutsvalLc);

    cutsLctopKpi->SetUsePID(true);
    AliAODPidHF* pidObjp=new AliAODPidHF();
    AliAODPidHF* pidObjK=new AliAODPidHF();
    AliAODPidHF* pidObjpi=new AliAODPidHF();
    pidObjp->SetMatch(1);
    pidObjK->SetMatch(1);
    pidObjpi->SetMatch(1);
    pidObjp->SetTPC(true);
    pidObjK->SetTPC(true);
    pidObjpi->SetTPC(true);
    pidObjp->SetTOF(true);
    pidObjK->SetTOF(true);
    pidObjpi->SetTOF(true);
    cutsLctopKpi->SetPidprot(pidObjp);
    cutsLctopKpi->SetPidHF(pidObjK);
    cutsLctopKpi->SetPidpion(pidObjpi);
    SetupCombinedPID(cutsLctopKpi,0.);
    cutsLctopKpi->SetPIDStrategy(AliRDHFCutsLctopKpi::kCombinedpPb);

    cutsLctopKpi->PrintAll();

    TFile outfile(outfilename.Data(), "recreate");
    cutsD0toKpi->Write();
    cutsDplustoKpipi->Write();
    cutsDstoKKpi->Write();
    cutsLctopKpi->Write();
    outfile.Close();
}

void CreateCutObjectsVeryLoose()
{
    AliESDtrackCuts* esdTrackCuts = new AliESDtrackCuts();
    esdTrackCuts->SetRequireSigmaToVertex(false);
    esdTrackCuts->SetRequireTPCRefit(true);
    esdTrackCuts->SetRequireITSRefit(true);
    esdTrackCuts->SetClusterRequirementITS(AliESDtrackCuts::kSPD, AliESDtrackCuts::kAny);
    esdTrackCuts->SetEtaRange(-0.8, 0.8);
    esdTrackCuts->SetMinDCAToVertexXY(0.);
    esdTrackCuts->SetPtRange(0.3, 1.e10);
    esdTrackCuts->SetMaxDCAToVertexXY(1.e6);

    const int nptbins = 2;
    float* ptbins;
    ptbins = new float[nptbins+1];
    ptbins[0]=0.;
    ptbins[1]=8.;
    ptbins[2]=100.;

    //D0_____________________________________________________________________________________
    const int nvarsD0 = 11;
    AliRDHFCutsD0toKpi* cutsD0toKpi=new AliRDHFCutsD0toKpi();
    cutsD0toKpi->SetName("D0toKpiCuts");
    cutsD0toKpi->AddTrackCuts(esdTrackCuts);
    cutsD0toKpi->SetPtBins(nptbins+1, ptbins);
    cutsD0toKpi->SetGlobalIndex(nvarsD0, nptbins);

    //m    dca      cost*  ptk ptpi  d0k          d0pi       d0d0          cosp  cosxy normdxy
    float cutsMatrixD0toKpiStand[nptbins][nvarsD0]=  {
        {0.300, 300.*1E-4, 1.0, 0.3, 0.3, 1000.*1E-4, 1000.*1E-4, 100000. *1E-8, 0.7, 0., 0.},
        {0.300, 300.*1E-4, 1.0, 0.3, 0.3, 1000.*1E-4, 1000.*1E-4, 100000. *1E-8, 0.7, 0., 0.}};

    float **cutsMatrixTransposeStand = new float*[nvarsD0];
    for(int iv=0;iv<nvarsD0;iv++)
        cutsMatrixTransposeStand[iv] = new float[nptbins];

    for (int ibin=0;ibin<nptbins;ibin++)
    {
        for (int ivar = 0; ivar<nvarsD0; ivar++)
        {
            cutsMatrixTransposeStand[ivar][ibin]=cutsMatrixD0toKpiStand[ibin][ivar];
        }
    }

    cutsD0toKpi->SetCuts(nvarsD0, nptbins, cutsMatrixTransposeStand);
    cutsD0toKpi->SetUseSpecialCuts(false);
    for(int iv = 0; iv < nvarsD0; iv++)
        delete [] cutsMatrixTransposeStand[iv];
    delete [] cutsMatrixTransposeStand;
    cutsMatrixTransposeStand = NULL;

    //pid settings
    cutsD0toKpi->SetUsePID(false);
    cutsD0toKpi->PrintAll();

    //D+_____________________________________________________________________________________
    AliRDHFCutsDplustoKpipi* cutsDplustoKpipi = new AliRDHFCutsDplustoKpipi();
    cutsDplustoKpipi->SetName("DplustoKpipiCuts");
    cutsDplustoKpipi->AddTrackCuts(esdTrackCuts);

    const int nvarsDplus=14;
    float** anacutsvalDplus;
    anacutsvalDplus=new float*[nvarsDplus];
    for(int ic=0; ic<nvarsDplus; ic++)
        anacutsvalDplus[ic] = new float[nptbins];

    anacutsvalDplus[0][0]=0.2;    //minv
    anacutsvalDplus[1][0]=0.3;    //ptK
    anacutsvalDplus[2][0]=0.3;    //ptpi
    anacutsvalDplus[3][0]=0.;     //d0K
    anacutsvalDplus[4][0]=0.;     //d0pi
    anacutsvalDplus[5][0]=0.;     //dist12
    anacutsvalDplus[6][0]=0.030;  //sigvert
    anacutsvalDplus[7][0]=0.03;   //decay length
    anacutsvalDplus[8][0]=0.0;    //pM
    anacutsvalDplus[9][0]=0.7;   //cosp
    anacutsvalDplus[10][0]=0.0;   //sumd02
    anacutsvalDplus[11][0]=1.e10; //dca
    anacutsvalDplus[12][0]=0.;    //norm dec len xy
    anacutsvalDplus[13][0]=0.0;  //cospXY

    anacutsvalDplus[0][1]=0.2;    //minv
    anacutsvalDplus[1][1]=0.3;    //ptK
    anacutsvalDplus[2][1]=0.3;    //ptpi
    anacutsvalDplus[3][1]=0.;     //d0K
    anacutsvalDplus[4][1]=0.;     //d0pi
    anacutsvalDplus[5][1]=0.;     //dist12
    anacutsvalDplus[6][1]=0.060;  //sigvert
    anacutsvalDplus[7][1]=0.03;   //decay length
    anacutsvalDplus[8][1]=0.0;    //pM
    anacutsvalDplus[9][1]=0.7;   //cosp
    anacutsvalDplus[10][1]=0.0;   //sumd02
    anacutsvalDplus[11][1]=1.e10; //dca
    anacutsvalDplus[12][1]=0.;    //norm dec len xy
    anacutsvalDplus[13][1]=0.00;  //cospXY

    cutsDplustoKpipi->SetPtBins(nptbins+1,ptbins);
    cutsDplustoKpipi->SetCuts(nvarsDplus,nptbins,anacutsvalDplus);
    cutsDplustoKpipi->SetScaleNormDLxyBypOverPt(false);
    cutsDplustoKpipi->SetUseImpParProdCorrCut(false);
    cutsDplustoKpipi->SetUsePID(false);
    cutsDplustoKpipi->SetMinPtCandidate(0.);
    cutsDplustoKpipi->SetMaxPtCandidate(100.);

    cutsDplustoKpipi->PrintAll();

    //Ds+_____________________________________________________________________________________
    AliRDHFCutsDstoKKpi* cutsDstoKKpi = new AliRDHFCutsDstoKKpi();
    cutsDstoKKpi->SetName("DstoKKpiCuts");
    cutsDstoKKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsDs = 20;
    float** anacutsvalDs;
    anacutsvalDs=new float*[nvarsDs];
    for(int ic=0;ic<nvarsDs;ic++)
        anacutsvalDs[ic]=new float[nptbins];

    anacutsvalDs[0][0]=0.35; //0 inv. mass [GeV]
    anacutsvalDs[1][0]=0.3; //1 pTK [GeV/c]
    anacutsvalDs[2][0]=0.3; //2 pTPi [GeV/c]
    anacutsvalDs[3][0]=0.; //3 d0K [cm]
    anacutsvalDs[4][0]=0.; //4 d0pi [cm]
    anacutsvalDs[5][0]=0.; //5 dist12 [cm]
    anacutsvalDs[6][0]=0.03; //6 sigmavert [cm]
    anacutsvalDs[7][0]=0.02; //7 decLen [cm]",
    anacutsvalDs[8][0]=0.; //8 ptMax [GeV/c]
    anacutsvalDs[9][0]=0.7; //9 cosThetaPoint
    anacutsvalDs[10][0]=0.; //10 Sum d0^2 (cm^2)
    anacutsvalDs[11][0]=1000.; //11 dca [cm]
    anacutsvalDs[12][0]=0.020; //12 inv. mass (Mphi-MKK) [GeV]
    anacutsvalDs[13][0]=0.2; //13 inv. mass (MKo*-MKpi) [GeV]
    anacutsvalDs[14][0]=0.; //14 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[15][0]=1.; //15 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[16][0]=0.; //16 decLenXY [cm]
    anacutsvalDs[17][0]=0.; //17 NormdecLen
    anacutsvalDs[18][0]=0.; //18 NormdecLenXY
    anacutsvalDs[19][0]=0.; //19 cosThetaPointXY

    anacutsvalDs[0][1]=0.35; //0 inv. mass [GeV]
    anacutsvalDs[1][1]=0.3; //1 pTK [GeV/c]
    anacutsvalDs[2][1]=0.3; //2 pTPi [GeV/c]
    anacutsvalDs[3][1]=0.; //3 d0K [cm]
    anacutsvalDs[4][1]=0.; //4 d0pi [cm]
    anacutsvalDs[5][1]=0.; //5 dist12 [cm]
    anacutsvalDs[6][1]=0.06; //6 sigmavert [cm]
    anacutsvalDs[7][1]=0.02; //7 decLen [cm]",
    anacutsvalDs[8][1]=0.; //8 ptMax [GeV/c]
    anacutsvalDs[9][1]=0.7; //9 cosThetaPoint
    anacutsvalDs[10][1]=0.; //10 Sum d0^2 (cm^2)
    anacutsvalDs[11][1]=1000.; //11 dca [cm]
    anacutsvalDs[12][1]=0.020; //12 inv. mass (Mphi-MKK) [GeV]
    anacutsvalDs[13][1]=0.2; //13 inv. mass (MKo*-MKpi) [GeV]
    anacutsvalDs[14][1]=0.; //14 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[15][1]=1.; //15 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[16][1]=0.; //16 decLenXY [cm]
    anacutsvalDs[17][1]=0.; //17 NormdecLen
    anacutsvalDs[18][1]=0.; //18 NormdecLenXY
    anacutsvalDs[19][1]=0.; //19 cosThetaPointXY

    cutsDstoKKpi->SetPtBins(nptbins+1,ptbins);
    cutsDstoKKpi->SetCuts(nvarsDs,nptbins,anacutsvalDs);
    cutsDstoKKpi->SetUsePID(false);
    cutsDstoKKpi->SetPidOption(0); //0=kConservative,1=kStrong
    cutsDstoKKpi->SetMinPtCandidate(0.);
    cutsDstoKKpi->SetMaxPtCandidate(100.);

    cutsDstoKKpi->PrintAll();

    //Lc+_____________________________________________________________________________________
    AliRDHFCutsLctopKpi* cutsLctopKpi=new AliRDHFCutsLctopKpi();
    cutsLctopKpi->SetName("LctopKpiCuts");
    cutsLctopKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsLc = 13;
    float** anacutsvalLc;
    anacutsvalLc=new float*[nvarsDs];
    for(int ic=0;ic<nvarsLc;ic++)
        anacutsvalLc[ic]=new float[nptbins];

    anacutsvalLc[0][0] = 0.18;
    anacutsvalLc[1][0] = 0.3;
    anacutsvalLc[2][0] = 0.3;
    anacutsvalLc[3][0] = 0.;
    anacutsvalLc[4][0] = 0.;
    anacutsvalLc[5][0] = 0.;
    anacutsvalLc[6][0] = 0.03;
    anacutsvalLc[7][0] = 0.;
    anacutsvalLc[8][0] = 0.;
    anacutsvalLc[9][0] = 0.7;
    anacutsvalLc[10][0] = 0.;
    anacutsvalLc[11][0] = 0.;
    anacutsvalLc[12][0] = 0.;

    anacutsvalLc[0][1] = 0.18;
    anacutsvalLc[1][1] = 0.3;
    anacutsvalLc[2][1] = 0.3;
    anacutsvalLc[3][1] = 0.;
    anacutsvalLc[4][1] = 0.;
    anacutsvalLc[5][1] = 0.;
    anacutsvalLc[6][1] = 0.06;
    anacutsvalLc[7][1] = 0.;
    anacutsvalLc[8][1] = 0.;
    anacutsvalLc[9][1] = 0.7;
    anacutsvalLc[10][1] = 0.;
    anacutsvalLc[11][1] = 0.;
    anacutsvalLc[12][1] = 0.;

    cutsLctopKpi->SetUseTrackSelectionWithFilterBits(true);
    cutsLctopKpi->SetPtBins(nptbins+1,ptbins);
    cutsLctopKpi->SetCuts(nvarsLc,nptbins,anacutsvalLc);

    cutsLctopKpi->SetUsePID(false);
    cutsLctopKpi->PrintAll();

    TFile outfile("D0DplusDsLcVeryLoosePreselectionCuts.root", "recreate");
    cutsD0toKpi->Write();
    cutsDplustoKpipi->Write();
    cutsDstoKKpi->Write();
    cutsLctopKpi->Write();
    outfile.Close();
}

void CreateCutObjectsPIDonly(TString outfilename = "D0DplusDsLcPreselectionCuts_onlyPID.root")
{
    AliESDtrackCuts* esdTrackCuts = new AliESDtrackCuts();
    esdTrackCuts->SetRequireSigmaToVertex(false);
    esdTrackCuts->SetRequireTPCRefit(true);
    esdTrackCuts->SetRequireITSRefit(true);
    esdTrackCuts->SetClusterRequirementITS(AliESDtrackCuts::kSPD, AliESDtrackCuts::kAny);
    esdTrackCuts->SetEtaRange(-0.8, 0.8);
    esdTrackCuts->SetMinDCAToVertexXY(0.);
    esdTrackCuts->SetPtRange(0.3, 1.e10);
    esdTrackCuts->SetMaxDCAToVertexXY(1.e6);

    const int nptbins = 1;
    float* ptbins;
    ptbins = new float[nptbins+1];
    ptbins[0]=0.;
    ptbins[1]=100.;

    //D0_____________________________________________________________________________________
    const int nvarsD0 = 11;
    AliRDHFCutsD0toKpi* cutsD0toKpi=new AliRDHFCutsD0toKpi();
    cutsD0toKpi->SetName("D0toKpiCuts");
    cutsD0toKpi->AddTrackCuts(esdTrackCuts);
    cutsD0toKpi->SetPtBins(nptbins+1, ptbins);
    cutsD0toKpi->SetGlobalIndex(nvarsD0, nptbins);

    //m    dca      cost*  ptk ptpi  d0k          d0pi       d0d0          cosp  cosxy normdxy
    float cutsMatrixD0toKpiStand[nptbins][nvarsD0]=  {{0.300, 300.*1E-4, 1.0, 0.3, 0.3, 1000.*1E-4, 1000.*1E-4, 100000. *1E-8, 0.,  0., 0.}};

    float **cutsMatrixTransposeStand = new float*[nvarsD0];
    for(int iv=0;iv<nvarsD0;iv++)
        cutsMatrixTransposeStand[iv] = new float[nptbins];

    for (int ibin=0;ibin<nptbins;ibin++)
    {
        for (int ivar = 0; ivar<nvarsD0; ivar++)
        {
            cutsMatrixTransposeStand[ivar][ibin]=cutsMatrixD0toKpiStand[ibin][ivar];
        }
    }

    cutsD0toKpi->SetCuts(nvarsD0, nptbins, cutsMatrixTransposeStand);
    cutsD0toKpi->SetUseSpecialCuts(false);
    for(int iv = 0; iv < nvarsD0; iv++)
        delete [] cutsMatrixTransposeStand[iv];
    delete [] cutsMatrixTransposeStand;
    cutsMatrixTransposeStand = NULL;

    //pid settings
    AliAODPidHF* pidObj = new AliAODPidHF();
    int mode = 1;
    const int nlims = 2;
    double plims[nlims] = {0.6, 0.8}; //TPC limits in momentum [GeV/c]
    bool compat = true; //effective only for this mode
    bool asym = true;
    double sigmas[5] = {2., 1., 0., 3., 0.}; //to be checked and to be modified with new implementation of setters by Rossella
    pidObj->SetAsym(asym);// if you want to use the asymmetric bands in TPC
    pidObj->SetMatch(mode);
    pidObj->SetPLimit(plims,nlims);
    pidObj->SetSigma(sigmas);
    pidObj->SetCompat(compat);
    pidObj->SetTPC(true);
    pidObj->SetTOF(true);
    pidObj->SetPCompatTOF(1.5);
    pidObj->SetSigmaForTPCCompat(3.);
    pidObj->SetSigmaForTOFCompat(3.);
    pidObj->SetOldPid(false);
    cutsD0toKpi->SetPidHF(pidObj);
    cutsD0toKpi->SetUsePID(true);
    cutsD0toKpi->SetMinPtCandidate(0.);
    cutsD0toKpi->SetMaxPtCandidate(100.);

    cutsD0toKpi->PrintAll();

    //D+_____________________________________________________________________________________
    AliRDHFCutsDplustoKpipi* cutsDplustoKpipi = new AliRDHFCutsDplustoKpipi();
    cutsDplustoKpipi->SetName("DplustoKpipiCuts");
    cutsDplustoKpipi->AddTrackCuts(esdTrackCuts);

    const int nvarsDplus=14;
    float** anacutsvalDplus;
    anacutsvalDplus=new float*[nvarsDplus];
    for(int ic=0; ic<nvarsDplus; ic++)
        anacutsvalDplus[ic] = new float[nptbins];

    anacutsvalDplus[0][0]=0.2;    //minv
    anacutsvalDplus[1][0]=0.3;    //ptK
    anacutsvalDplus[2][0]=0.3;    //ptpi
    anacutsvalDplus[3][0]=0.;     //d0K
    anacutsvalDplus[4][0]=0.;     //d0pi
    anacutsvalDplus[5][0]=0.;     //dist12
    anacutsvalDplus[6][0]=0.06;  //sigvert
    anacutsvalDplus[7][0]=0.;   //decay length
    anacutsvalDplus[8][0]=0.;    //pM
    anacutsvalDplus[9][0]=0.;   //cosp
    anacutsvalDplus[10][0]=0.0;   //sumd02
    anacutsvalDplus[11][0]=1.e10; //dca
    anacutsvalDplus[12][0]=0.;    //norm dec len xy
    anacutsvalDplus[13][0]=0.;  //cospXY

    cutsDplustoKpipi->SetPtBins(nptbins+1,ptbins);
    cutsDplustoKpipi->SetCuts(nvarsDplus,nptbins,anacutsvalDplus);
    cutsDplustoKpipi->SetScaleNormDLxyBypOverPt(false);
    cutsDplustoKpipi->SetUseImpParProdCorrCut(false);
    cutsDplustoKpipi->SetUsePID(true);
    cutsDplustoKpipi->SetMinPtCandidate(0.);
    cutsDplustoKpipi->SetMaxPtCandidate(100.);

    cutsDplustoKpipi->PrintAll();

    //Ds+_____________________________________________________________________________________
    AliRDHFCutsDstoKKpi* cutsDstoKKpi = new AliRDHFCutsDstoKKpi();
    cutsDstoKKpi->SetName("DstoKKpiCuts");
    cutsDstoKKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsDs = 20;
    float** anacutsvalDs;
    anacutsvalDs=new float*[nvarsDs];
    for(int ic=0;ic<nvarsDs;ic++)
        anacutsvalDs[ic]=new float[nptbins];

    anacutsvalDs[0][0]=0.35; //0 inv. mass [GeV]
    anacutsvalDs[1][0]=0.3; //1 pTK [GeV/c]
    anacutsvalDs[2][0]=0.3; //2 pTPi [GeV/c]
    anacutsvalDs[3][0]=0.; //3 d0K [cm]
    anacutsvalDs[4][0]=0.; //4 d0pi [cm]
    anacutsvalDs[5][0]=0.; //5 dist12 [cm]
    anacutsvalDs[6][0]=0.06; //6 sigmavert [cm]
    anacutsvalDs[7][0]=0.; //7 decLen [cm]",
    anacutsvalDs[8][0]=0.; //8 ptMax [GeV/c]
    anacutsvalDs[9][0]=0.; //9 cosThetaPoint
    anacutsvalDs[10][0]=0.; //10 Sum d0^2 (cm^2)
    anacutsvalDs[11][0]=1000.; //11 dca [cm]
    anacutsvalDs[12][0]=0.020; //12 inv. mass (Mphi-MKK) [GeV]
    anacutsvalDs[13][0]=0.2; //13 inv. mass (MKo*-MKpi) [GeV]
    anacutsvalDs[14][0]=0.; //14 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[15][0]=1.; //15 Abs(CosineKpiPhiRFrame)^3
    anacutsvalDs[16][0]=0.; //16 decLenXY [cm]
    anacutsvalDs[17][0]=0.; //17 NormdecLen
    anacutsvalDs[18][0]=0.; //18 NormdecLenXY
    anacutsvalDs[19][0]=0.; //19 cosThetaPointXY

    cutsDstoKKpi->SetPtBins(nptbins+1,ptbins);
    cutsDstoKKpi->SetCuts(nvarsDs,nptbins,anacutsvalDs);
    cutsDstoKKpi->SetUsePID(true);
    cutsDstoKKpi->SetPidOption(0); //0=kConservative,1=kStrong
    cutsDstoKKpi->SetMinPtCandidate(0.);
    cutsDstoKKpi->SetMaxPtCandidate(100.);

    cutsDstoKKpi->PrintAll();

    //Lc+_____________________________________________________________________________________
    AliRDHFCutsLctopKpi* cutsLctopKpi=new AliRDHFCutsLctopKpi();
    cutsLctopKpi->SetName("LctopKpiCuts");
    cutsLctopKpi->AddTrackCuts(esdTrackCuts);

    const int nvarsLc = 13;
    float** anacutsvalLc;
    anacutsvalLc=new float*[nvarsDs];
    for(int ic=0;ic<nvarsLc;ic++)
        anacutsvalLc[ic]=new float[nptbins];

    anacutsvalLc[0][0] = 0.18;
    anacutsvalLc[1][0] = 0.3;
    anacutsvalLc[2][0] = 0.3;
    anacutsvalLc[3][0] = 0.;
    anacutsvalLc[4][0] = 0.;
    anacutsvalLc[5][0] = 0.;
    anacutsvalLc[6][0] = 0.06;
    anacutsvalLc[7][0] = 0.;
    anacutsvalLc[8][0] = 0.;
    anacutsvalLc[9][0] = -1.;
    anacutsvalLc[10][0] = 0.;
    anacutsvalLc[11][0] = 0.05;
    anacutsvalLc[12][0] = 0.3;

    cutsLctopKpi->SetUseTrackSelectionWithFilterBits(true);
    cutsLctopKpi->SetPtBins(nptbins+1,ptbins);
    cutsLctopKpi->SetCuts(nvarsLc,nptbins,anacutsvalLc);

    cutsLctopKpi->SetUsePID(true);
    AliAODPidHF* pidObjp=new AliAODPidHF();
    AliAODPidHF* pidObjK=new AliAODPidHF();
    AliAODPidHF* pidObjpi=new AliAODPidHF();
    pidObjp->SetMatch(1);
    pidObjK->SetMatch(1);
    pidObjpi->SetMatch(1);
    pidObjp->SetTPC(true);
    pidObjK->SetTPC(true);
    pidObjpi->SetTPC(true);
    pidObjp->SetTOF(true);
    pidObjK->SetTOF(true);
    pidObjpi->SetTOF(true);
    cutsLctopKpi->SetPidprot(pidObjp);
    cutsLctopKpi->SetPidHF(pidObjK);
    cutsLctopKpi->SetPidpion(pidObjpi);
    SetupCombinedPID(cutsLctopKpi,0.);
    cutsLctopKpi->SetPIDStrategy(AliRDHFCutsLctopKpi::kCombinedpPb);

    cutsLctopKpi->PrintAll();

    TFile outfile(outfilename.Data(), "recreate");
    cutsD0toKpi->Write();
    cutsDplustoKpipi->Write();
    cutsDstoKKpi->Write();
    cutsLctopKpi->Write();
    outfile.Close();
}
