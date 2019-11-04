#!/bin/bash

arg=$1
filename=$2


#This means that filename will go get the first argument in the "argument" line in the script that does the Condor submissions, the doCondor.py

currentDir=/home/mhadley/preprocessing5GeVUpsilonSamples/CMSSW_10_2_15/src/low_pt_tau_reco/preprocessing

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=slc7_amd64_gcc700

cd /home/mhadley/preprocessing5GeVUpsilonSamples/CMSSW_10_2_15/src/

eval `scramv1 runtime -sh` #This is what cmsenv actually is, thank goodness for aliases

cd $currentDir

python ${currentDir}/LowPtTauMatcher.py  inputFiles=$arg suffix=$filename
