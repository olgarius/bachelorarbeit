import numpy as np
import numpy.lib.recfunctions as rf   

from pathlib import Path

import util

relevantKeys = ['bjet2_e','bjet1_e','bjet2_eta','bjet2_phi','bjet1_eta','bjet1_phi','bjet2_pt','bjet1_pt','genBQuark2_e','genBQuark1_e','genBQuark2_eta','genBQuark2_phi','genBQuark1_eta','genBQuark1_phi','genBQuark2_pt','genBQuark1_pt']

DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')
data = np.load(DATAPATH)
events = data['events']

relevantData = events[relevantKeys]

repackedData = rf.repack_fields(relevantData)

new_bjet1e, new_bjet2e, new_bjet1pt, new_bjet2pt, new_bjet1phi, new_bjet2phi, new_bjet1eta, new_bjet2eta, notusabeleEvents = util.matchBjets(repackedData,'bjet1_e','bjet2_e','bjet1_pt','bjet2_pt','bjet1_phi','bjet2_phi','genBQuark1_phi','genBQuark2_phi','bjet1_eta','bjet2_eta','genBQuark1_eta','genBQuark2_eta',)

updatedData1_1 = util.updateStructArray(repackedData, 'bjet1_e', new_bjet1e)
updatedData1_2 = util.updateStructArray(updatedData1_1, 'bjet2_e', new_bjet2e)
updatedData1_3 = util.updateStructArray(updatedData1_2, 'bjet1_pt', new_bjet1pt)
updatedData1_4 = util.updateStructArray(updatedData1_3, 'bjet2_pt', new_bjet2pt)
updatedData1_5 = util.updateStructArray(updatedData1_4, 'bjet1_phi', new_bjet1phi)
updatedData1_6 = util.updateStructArray(updatedData1_5, 'bjet2_phi', new_bjet2phi)
updatedData1_7 = util.updateStructArray(updatedData1_6, 'bjet1_eta', new_bjet1eta)
updatedData1_8 = util.updateStructArray(updatedData1_7, 'bjet2_eta', new_bjet2eta)





notusabeleEvents += util.getIndexForCondition(updatedData1_8, 'bjet1_pt', np.less, 15)
notusabeleEvents += util.getIndexForCondition(updatedData1_8, 'bjet2_pt', np.less, 15)

print('removed ', len(set(notusabeleEvents)), ' Events')
updatedData2_1 = util.removeEvents(updatedData1_8,notusabeleEvents)




table = util.readTable('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/Pythia17_MC_PtResolution_AK4PFchs.txt')

bjet1sigma = util.getSigmas(updatedData2_1,'bjet1_pt','bjet1_eta',table)
bjet2sigma = util.getSigmas(updatedData2_1,'bjet2_pt','bjet2_eta',table)

updatedData3_1 = util.updateStructArray(updatedData2_1, 'bjet1_sigma',bjet1sigma)
updatedData3_2 = util.updateStructArray(updatedData3_1, 'bjet2_sigma',bjet2sigma)

facs = util.getFacs(updatedData3_2,'bjet1_phi','bjet1_eta','bjet2_phi','bjet2_eta')

updatedData4_1 = util.updateStructArray(updatedData3_2,'bie_to_bje', facs)


np.save('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet',updatedData4_1)



