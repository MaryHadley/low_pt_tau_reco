#!/usr/bin/env python
import os, sys, glob

inputFileDir = '/home/mhadley/preprocessing5GeVUpsilonSamples/CMSSW_10_2_15/src/low_pt_tau_reco/preprocessing/UpsilonToTauTau_3prong_m5'
inputFileName = 'UpsilonToTauTau_PUPoissonAve20_102X_upgrade2018_realistic_v18_3prong_m5_miniaod_'
inputFileList = glob.glob('%s/%s*.root' %(inputFileDir,  inputFileName))
print 'inputFileList is:', inputFileList

outDir = inputFileDir.replace('/', '-')[1:]
print outDir
#Note if you ask makedirs to make more than one directory, it won't work, so for example here I had to make condor_outputs by hand and then it was able to make the next directory I requested...basically can only make one directory, not one and then another inside it and another inside that blah blah
def condor():
    basedir = os.getcwd()
    if not os.path.exists(basedir + '/condor_outputs/%s' %(inputFileDir.replace('/', '-')[1:])):
        os.makedirs(basedir + '/condor_outputs/%s' %(inputFileDir.replace('/', '-')[1:]))
    for file in inputFileList:
        print 'file is:', file

        jdfFileName = '%s.job' %file.split('/')[-1][:-5]
        print 'jdfFileName is:', jdfFileName
        dict = {'dir': basedir, 'filename': jdfFileName[:-4], 'newDir': outDir, 'arg': file }
        #filename is actually both the path to the file and its name, but I am too tired to change this to a better name right now
        jdf = open(jdfFileName, 'w')
        jdf.write("""

universe = vanilla
executable = %(dir)s/doCondor.sh
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Notification = Error

use_x509userproxy=true

Arguments = %(arg)s %(filename)s
Output = %(dir)s/condor_outputs/%(newDir)s/log_%(filename)s.stdout
Error = %(dir)s/condor_outputs/%(newDir)s/log_%(filename)s.stderr
Log = %(dir)s/condor_outputs/%(newDir)s/log_%(filename)s.condorlog

Queue 1

        """%dict)
        jdf.close()
        os.system('condor_submit %s' %jdfFileName) #commented out for testing
pass
condor()


