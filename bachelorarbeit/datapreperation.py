import numpy as np
import numpy.lib.recfunctions as rf   
from numpy.linalg import inv

from pathlib import Path
from glob import glob 

import mathutil as mu
import structutil as su
import datautil as du

from PATHS import RAWDATAPATH

relevantKeys = ['bjet2_e','bjet1_e','bjet2_eta','bjet2_phi','bjet1_eta','bjet1_phi','bjet2_pt','bjet1_pt','genBQuark2_e','genBQuark1_e','genBQuark2_eta','genBQuark2_phi','genBQuark1_eta','genBQuark1_phi','genBQuark2_pt','genBQuark1_pt','rho','nbjetscand','bjet1_btag_deepFlavor','bjet2_btag_deepFlavor','dau1_eta','dau2_eta', 'dau1_phi','dau2_phi' ,'genLepton1_eta','genLepton2_eta', 'genLepton1_phi','genLepton2_phi','tauH_mass','dau2_e','dau1_e','dau2_pt','dau1_pt','genLepton1_pt','genLepton2_e','genLepton1_e','genLepton2_pt', 'met_cov00', 'met_cov01','met_cov11','pairType', 'met_et', 'met_phi','genNu1_e','genNu2_e','genNu1_pt','genNu2_pt','genNu1_phi','genNu2_phi','genNu1_eta','genNu2_eta' ]


files = glob(RAWDATAPATH+'*.npz')

for i,f in enumerate(files):
    data = np.load(f)
    if i is 0:
        events = data['events']
    else:
        events = np.append(events,data['events'])

# path = RAWDATAPATH + 'tet.npz'

# events =  np.load(path)['events']

print('Size of rawdata:', len(events))

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
TauSwitchList = (('dau2_e','dau1_e'),('dau2_eta','dau1_eta'),('dau2_phi','dau1_phi'),('dau2_pt','dau1_pt'),('genNu1_e','genNu2_e'),('genNu1_pt','genNu2_pt'),('genNu1_phi','genNu2_phi'),('genNu1_eta','genNu2_eta'))



dataSet, notuseableEvents = du.deltaRmatching(repackedData, 'bjet1_eta','bjet2_eta', 'bjet1_phi','bjet2_phi' ,'genBQuark1_eta','genBQuark2_eta', 'genBQuark1_phi','genBQuark2_phi', BquarkSwitchList )
dataSet, notuseableEvents2 = du.deltaRmatching(dataSet, 'dau1_eta','dau2_eta', 'dau1_phi','dau2_phi' ,'genLepton1_eta','genLepton2_eta', 'genLepton1_phi','genLepton2_phi', TauSwitchList )

print('no match delta R b: ',len(notuseableEvents), len(notuseableEvents)/len(events)*100, '%')
print('no match delta R tau: ',len(notuseableEvents2),len(notuseableEvents2)/len(events)*100, '%')


notuseableEvents += notuseableEvents2
# print(set(notuseableEvents1) == set(notuseableEvents))


# notuseableEvents += util.getIndexForCondition(dataSet, 'bjet1_pt', np.less, 15)
# notuseableEvents += util.getIndexForCondition(dataSet, 'bjet2_pt', np.less, 15)

# notuseableEvents += su.getIndexForCondition(dataSet, 'genBQuark1_pt', np.less, 40)
# notuseableEvents += su.getIndexForCondition(dataSet, 'genBQuark2_pt', np.less, 40)

l0 = len(notuseableEvents)
notuseableEvents += su.getIndexForCondition(dataSet, 'nbjetscand', np.less_equal, 1)
l1 = len(notuseableEvents) - l0
notuseableEvents += su.getIndexForCondition(dataSet, 'bjet1_btag_deepFlavor', np.less, 0.304)
l2 = len(notuseableEvents) - l0 -l1
notuseableEvents += su.getIndexForCondition(dataSet, 'bjet2_btag_deepFlavor', np.less, 0.304)
l3 = len(notuseableEvents) - l0 -l1 - l2
notuseableEvents += su.getIndexForCondition(dataSet, 'pairType', np.greater, 2)
l4 = len(notuseableEvents) - l0 -l1 - l2 -l3


print('not enough b jets: ', l1,l1/len(events)*100, '%')
print('not sure enough b jets 1: ', l2,l2/len(events)*100, '%')
print('not sure enough b jets 2: ', l3,l3/len(events)*100, '%')
print('to many neutrinos: ', l4,l4/len(events)*100, '%')
print(len(set(notuseableEvents)), len(set(notuseableEvents))/len(events)*100, '%')
dataSet = su.removeEvents(dataSet,notuseableEvents)




table = du.readTable('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/Pythia17_MC_PtResolution_AK4PFchs.txt')

bjet1sigma = du.getSigmas(dataSet,'bjet1_pt','bjet1_eta',table, rhokey='rho')
bjet2sigma = du.getSigmas(dataSet,'bjet2_pt','bjet2_eta',table, rhokey='rho')

dataSet = su.updateStructArray(dataSet, 'bjet1_sigma',bjet1sigma)
dataSet = su.updateStructArray(dataSet, 'bjet2_sigma',bjet2sigma)

facs = du.getFacs(dataSet,'bjet1_pt','bjet1_phi','bjet1_eta','bjet2_pt','bjet2_phi','bjet2_eta')
genfacs = du.getFacs(dataSet,'genBQuark1_pt','genBQuark1_phi','genBQuark1_eta','genBQuark2_pt','genBQuark2_phi','genBQuark2_eta')

facstau = du.getFacs(dataSet,'dau1_pt','dau1_phi','dau1_eta','dau2_pt','dau2_phi','dau2_eta')
genfacstau = du.getFacs(dataSet,'genLepton1_pt','genLepton1_phi','genLepton1_eta','genLepton2_pt','genLepton2_phi','genLepton2_eta')


dataSet = su.updateStructArray(dataSet,'bie_to_bje', facs)
dataSet = su.updateStructArray(dataSet,'genbie_to_genbje', genfacs)

dataSet = su.updateStructArray(dataSet,'tauie_to_tauje', facstau)
dataSet = su.updateStructArray(dataSet,'gentauie_to_gentauje', genfacstau)

metX = np.cos(su.structToArray(dataSet, 'met_phi'))*su.structToArray(dataSet,'met_et')
metY = np.sin(su.structToArray(dataSet, 'met_phi'))*su.structToArray(dataSet,'met_et')

debugMetX = np.cos(su.structToArray(dataSet, 'genNu1_phi'))*su.structToArray(dataSet,'genNu1_pt')+np.cos(su.structToArray(dataSet, 'genNu2_phi'))*su.structToArray(dataSet,'genNu2_pt')
debugMetY = np.sin(su.structToArray(dataSet, 'genNu1_phi'))*su.structToArray(dataSet,'genNu1_pt')+np.sin(su.structToArray(dataSet, 'genNu2_phi'))*su.structToArray(dataSet,'genNu2_pt')


tau1_ex, tau1_ey, tau1_ez = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'dau1_pt'),su.structToArray(dataSet,'dau1_phi'),su.structToArray(dataSet,'dau1_eta')))
tau2_ex, tau2_ey, tau2_ez = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'dau2_pt'),su.structToArray(dataSet,'dau2_phi'),su.structToArray(dataSet,'dau2_eta')))

debugTau1_exComplete, debugTau1_eyComplete, debugTau1_ezComplete = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'genLepton1_pt'),su.structToArray(dataSet,'genLepton1_phi'),su.structToArray(dataSet,'genLepton1_eta')))
debugTau2_exComplete, debugTau2_eyComplete, debugTau2_ezComplete = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'genLepton2_pt'),su.structToArray(dataSet,'genLepton2_phi'),su.structToArray(dataSet,'genLepton2_eta')))

debugNu1_ex , debugNu1_ey , debugNu1_ez  = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'genNu1_pt'),su.structToArray(dataSet,'genNu1_phi'),su.structToArray(dataSet,'genNu1_eta')))
debugNu2_ex , debugNu2_ey , debugNu2_ez  = mu.detCoordinatesToCartesian((su.structToArray(dataSet,'genNu2_pt'),su.structToArray(dataSet,'genNu2_phi'),su.structToArray(dataSet,'genNu2_eta')))

debugTau1_ex = debugTau1_exComplete - debugNu1_ex
debugTau1_ey = debugTau1_eyComplete - debugNu1_ey
debugTau1_ez = debugTau1_ezComplete - debugNu1_ez

debugTau1_e = su.structToArray(dataSet,'genLepton1_e')-su.structToArray(dataSet,'genNu1_e')

debugTau2_ex = debugTau2_exComplete - debugNu2_ex
debugTau2_ey = debugTau2_eyComplete - debugNu2_ey
debugTau2_ez = debugTau2_ezComplete - debugNu2_ez

debugTau2_e = su.structToArray(dataSet,'genLepton2_e')-su.structToArray(dataSet,'genNu2_e')


dataSet = su.updateStructArray(dataSet,'dau1_ex',tau1_ex)
dataSet = su.updateStructArray(dataSet,'dau2_ex',tau2_ex)

dataSet = su.updateStructArray(dataSet,'dau1_ey',tau1_ey)
dataSet = su.updateStructArray(dataSet,'dau2_ey',tau2_ey)

dataSet = su.updateStructArray(dataSet,'dau1_ez',tau1_ez)
dataSet = su.updateStructArray(dataSet,'dau2_ez',tau2_ez)

dataSet = su.updateStructArray(dataSet,'met_x',metX)
dataSet = su.updateStructArray(dataSet,'met_y',metY)

dataSet = su.updateStructArray(dataSet,'debugdau1_ex',debugTau1_ex)
dataSet = su.updateStructArray(dataSet,'debugdau2_ex',debugTau2_ex)

dataSet = su.updateStructArray(dataSet,'debugdau1_ey',debugTau1_ey)
dataSet = su.updateStructArray(dataSet,'debugdau2_ey',debugTau2_ey)

dataSet = su.updateStructArray(dataSet,'debugdau1_ez',debugTau1_ez)
dataSet = su.updateStructArray(dataSet,'debugdau2_ez',debugTau2_ez)

dataSet = su.updateStructArray(dataSet,'debugdau1_e',debugTau1_e)
dataSet = su.updateStructArray(dataSet,'debugdau2_e',debugTau2_e)

dataSet = su.updateStructArray(dataSet,'debugmet_x',debugMetX)
dataSet = su.updateStructArray(dataSet,'debugmet_y',debugMetY)


inverseCovMat = np.array([inv([[a,b],[b,c]]) for a,b,c in zip(dataSet['met_cov00'], dataSet['met_cov01'], dataSet['met_cov11'])])
notuseableEvents3 = [i for i in range(len(inverseCovMat)) if inverseCovMat[i][0][0] == np.nan]

if len(notuseableEvents3) != 0:
    inverseCovMat = su.removeEvents(inverseCovMat,notuseableEvents3)



dataSet = su.removeEvents(dataSet,notuseableEvents3)
print('removed ', (len(set(notuseableEvents))+ len(notuseableEvents3))/len(repackedData)*100, '%')




inverseCovMat00 = np.array([a[0][0] for a in inverseCovMat])
inverseCovMat01 = np.array([a[0][1] for a in inverseCovMat])
inverseCovMat11 = np.array([a[1][1] for a in inverseCovMat])




dataSet = su.updateStructArray(dataSet, 'met_invcov00',inverseCovMat00)
dataSet = su.updateStructArray(dataSet, 'met_invcov01',inverseCovMat01)
dataSet = su.updateStructArray(dataSet, 'met_invcov11',inverseCovMat11)






print(len(inverseCovMat))

np.save('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet',dataSet)



