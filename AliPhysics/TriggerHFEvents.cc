#if !defined (__CINT__) || defined (__CLING__)

#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include <TChain.h>
#include <TGrid.h>
#include <TTree.h>
#include <TFile.h>
#include <TDirectoryFile.h>

#endif

void TriggerHFEvents(std::string cfgFile="config_trigger.yml")
{
    //load config
    YAML::Node config = YAML::LoadFile(cfgFile.data());
    std::string inFileName = config["infile"]["name"].as<std::string>();
    std::string inDirName = config["infile"]["dir"].as<std::string>();
    std::string outFileName = config["outfile"]["name"].as<std::string>();
    double ptMin2Prong = config["cuts"]["2prong"]["ptmin"].as<double>();

    // load input tree
    TFile* inFile = TFile::Open(inFileName.data());
    TDirectoryFile* inDir = static_cast<TDirectoryFile*>(inFile->Get(inDirName.data()));
    TTree* evTree = static_cast<TTree*>(inDir->Get("fRecoTree"));
    std::vector<Charm2Prong> *ch2ProngVec = nullptr;
    evTree->SetBranchAddress("Charm2Prong", &ch2ProngVec);

    // define output objects
    auto hTriggeredEvents = new TH1F("hTriggeredEvents", "", 2, 0.5, 2.5);
    hTriggeredEvents->GetXaxis()->SetBinLabel(1, "processed events");
    hTriggeredEvents->GetXaxis()->SetBinLabel(2, "2-prong triggered events");
    //TODO: add control plots (invariant mass, k* distribution etc)

    // loop over events
    int nEvents = 10000000; //evTree->GetEntriesFast();
    int nTriggeredEvents2Prong = 0;
    for(int iEv=0; iEv<nEvents; iEv++)
    {
        evTree->GetEvent(iEv);
        if(iEv%1000000==0)
            std::cout << Form("Processing event %d/%d", iEv, nEvents) << std::endl;

        // skip if there are no charm-hadron candidates
        if(ch2ProngVec->size() == 0)
            continue;

        //loop over 2-prong candidates
        int nSelected2Prong = 0;
        for(unsigned int i2Prong=0; i2Prong<ch2ProngVec->size(); i2Prong++)
        {
            float pt = ch2ProngVec->at(i2Prong).fPt;
            if(pt < ptMin2Prong)
                continue;
            //TODO: add more selections on topology
            nSelected2Prong++;

            //TODO: loop over tracks and compute relative momentum
        }

        if(nSelected2Prong>0)
            nTriggeredEvents2Prong++;
    }

    hTriggeredEvents->SetBinContent(1, nEvents);
    hTriggeredEvents->SetBinContent(2, nTriggeredEvents2Prong);

    // save output file
    TFile outFile(outFileName.data(), "recreate");
    hTriggeredEvents->Write();
    outFile.Close();
}