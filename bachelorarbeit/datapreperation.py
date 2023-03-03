import numpy as np
import numpy.lib.recfunctions as rf   

from pathlib import Path
from glob import glob 

import mathutil as mu
import structutil as su
import datautil as du

from PATHS import RAWDATAPATH

relevantKeys = ['bjet2_e','bjet1_e','bjet2_eta','bjet2_phi','bjet1_eta','bjet1_phi','bjet2_pt','bjet1_pt','genBQuark2_e','genBQuark1_e','genBQuark2_eta','genBQuark2_phi','genBQuark1_eta','genBQuark1_phi','genBQuark2_pt','genBQuark1_pt','rho','nbjetscand','bjet1_btag_deepFlavor','bjet2_btag_deepFlavor']


# files = glob(RAWDATAPATH+'*.npz')

# for i,f in enumerate(files):
#     data = np.load(f)
#     if i is 0:
#         events = data['events']
#     else:
#         events = np.append(events,data['events'])

path = RAWDATAPATH + 'tet.npz'

events =  np.load(path)['events']

print(len(events))

relevantData = events[relevantKeys]

repackedData = rf.repack_fields(relevantData)

indexarray = np.linspace(0,len(relevantData)-1,len(relevantData))

repackedData = su.updateStructArray(repackedData, 'index', indexarray)

# new_bjet1e, new_bjet2e, new_bjet1pt, new_bjet2pt, new_bjet1phi, new_bjet2phi, new_bjet1eta, new_bjet2eta, notuseableEvents1 = du.matchBjets(repackedData,'bjet1_e','bjet2_e','bjet1_pt','bjet2_pt','bjet1_phi','bjet2_phi','genBQuark1_phi','genBQuark2_phi','bjet1_eta','bjet2_eta','genBQuark1_eta','genBQuark2_eta',)

# dataSet = su.updateStructArray(repackedData, 'bjet1_e', new_bjet1e)
# dataSet = su.updateStructArray(dataSet, 'bjet2_e', new_bjet2e)
# dataSet = su.updateStructArray(dataSet, 'bjet1_pt', new_bjet1pt)
# dataSet = su.updateStructArray(dataSet, 'bjet2_pt', new_bjet2pt)
# dataSet = su.updateStructArray(dataSet, 'bjet1_phi', new_bjet1phi)
# dataSet = su.updateStructArray(dataSet, 'bjet2_phi', new_bjet2phi)
# dataSet = su.updateStructArray(dataSet, 'bjet1_eta', new_bjet1eta)
# dataSet = su.updateStructArray(dataSet, 'bjet2_eta', new_bjet2eta)


BquarkSwitchList = (('bjet2_e','bjet1_e'),('bjet2_eta','bjet1_eta'),('bjet2_phi','bjet1_phi'),('bjet2_pt','bjet1_pt'),('bjet1_btag_deepFlavor','bjet2_btag_deepFlavor'))
TauSwitchList = (('dau2_e','dau1_e'),('dau2_eta','dau1_eta'),('dau2_phi','dau1_phi'),('dau2_pt','dau1_pt'))



dataSet, notuseableEvents = du.deltaRmatching(repackedData, 'bjet1_eta','bjet2_eta', 'bjet1_phi','bjet2_phi' ,'genBQuark1_eta','genBQuark2_eta', 'genBQuark1_phi','genBQuark2_phi', BquarkSwitchList )

# print(set(notuseableEvents1) == set(notuseableEvents))


# notuseableEvents += util.getIndexForCondition(dataSet, 'bjet1_pt', np.less, 15)
# notuseableEvents += util.getIndexForCondition(dataSet, 'bjet2_pt', np.less, 15)

notuseableEvents += su.getIndexForCondition(dataSet, 'genBQuark1_pt', np.less, 40)
notuseableEvents += su.getIndexForCondition(dataSet, 'genBQuark2_pt', np.less, 40)
notuseableEvents += su.getIndexForCondition(dataSet, 'nbjetscand', np.less, 1)
notuseableEvents += su.getIndexForCondition(dataSet, 'bjet1_btag_deepFlavor', np.less, 0.304)
notuseableEvents += su.getIndexForCondition(dataSet, 'bjet2_btag_deepFlavor', np.less, 0.304)




print('removed ', len(set(notuseableEvents))/len(dataSet)*100, '%')
dataSet = su.removeEvents(dataSet,notuseableEvents)




table = du.readTable('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/Pythia17_MC_PtResolution_AK4PFchs.txt')

bjet1sigma = du.getSigmas(dataSet,'bjet1_pt','bjet1_eta',table, rhokey='rho')
bjet2sigma = du.getSigmas(dataSet,'bjet2_pt','bjet2_eta',table, rhokey='rho')

dataSet = su.updateStructArray(dataSet, 'bjet1_sigma',bjet1sigma)
dataSet = su.updateStructArray(dataSet, 'bjet2_sigma',bjet2sigma)

facs = du.getFacs(dataSet,'bjet1_pt','bjet1_phi','bjet1_eta','bjet2_pt','bjet2_phi','bjet2_eta')
genfacs = du.getFacs(dataSet,'genBQuark1_pt','genBQuark1_phi','genBQuark1_eta','genBQuark2_pt','genBQuark2_phi','genBQuark2_eta')



dataSet = su.updateStructArray(dataSet,'bie_to_bje', facs)
dataSet = su.updateStructArray(dataSet,'genbie_to_genbje', genfacs)



np.save('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet',dataSet)



