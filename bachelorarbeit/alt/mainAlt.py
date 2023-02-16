import grid
import bachelorarbeit.alt.functionGeneratorAlt as fG
import util as u
import functions as fun
import plotting as p


import numpy as np
import copy


from pathlib import Path

from multiprocessing import  Pool

DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')
KEYLIST = ['']

SIGMA_BE = 10
SIGMA_DAUE = 10



#load data and select relevant data
data = np.load(DATAPATH)

events = data['events']
#reldata = events[KEYLIST]

functions = fG.functionGenerator(events,SIGMA_BE,SIGMA_DAUE)
fdimss = functions.fdimss
temp = functions.getFunctionBatch()
fs = copy.deepcopy(temp)


evaArrays = [np.linspace(0,300,100),np.linspace(0,300,100)]

def processEvent(fs):
    g = grid.grid()

    for a in evaArrays:
        g.addDimension(a)
    
    for i,f in enumerate(fs):
        g.addFunction(f,fdimss[i])
    
    g.evaluate()

    return g.getMinCoords(), g.evaluated



# if __name__ == '__main__':
#     with Pool() as p:
#         results = p.starmap(processEvent, fs)
#     print('finished')
   


results = []

first = True

for f in fs:
    mincoords, g =processEvent(f)
    results.append(mincoords)


    if first:
        cg = g
       
        print(mincoords)
        first = False
    else:
        cg = cg + g

p.plot2d3(cg, evaArrays,'./', 'bE','dauE',[0,100],[0,100])



be2array = []


with open('testresultB.txt','w') as fi:

    H_mass = u.structToArray(events,'bH_mass')
    pBeta1 = u.structToArray(events,'bjet1_eta')
    pBeta2 = u.structToArray(events,'bjet2_eta')
    pBphi1 = u.structToArray(events,'bjet1_phi')
    pBphi2 = u.structToArray(events,'bjet2_phi')

    fi.write('E_1_fitted'+'\t'+'E_2(E_1_fitted)'+'\t'+'E_2(E_1_fitted)+E_1_fitted'+'\n')

    for i,l in enumerate(results):
        pB1 = (1.0,pBphi1[i],u.pseudoRapToPolar(pBeta1[i]))
        pB2 = (1.0,pBphi2[i],u.pseudoRapToPolar(pBeta2[i]))
        alpha = u.angleBetweenVectorsInPolarCoordinats(pB1,pB2)
        bE2 = fun.bEiofbEj(H_mass[i],l[0],alpha)
        be2array.append(bE2)
        fi.write(str(l[0])+'\t'+str(bE2)+'\t'+str(bE2+l[0])+'\n')
        

    
    
    fi.close()

daue2array = []

with open('testresultDau.txt','w') as fi:

    H_mass = u.structToArray(events,'tauH_mass')
    pDaueta1 = u.structToArray(events,'dau1_eta')
    pDaueta2 = u.structToArray(events,'dau2_eta')
    pDauphi1 = u.structToArray(events,'dau1_phi')
    pDauphi2 = u.structToArray(events,'dau2_phi')

    fi.write('E_1_fitted'+'\t'+'E_2(E_1_fitted)'+'\t'+'E_2(E_1_fitted)+E_1_fitted'+'\n')

    for i,l in enumerate(results):
        pDau1 = (1.0,pDauphi1[i],u.pseudoRapToPolar(pDaueta1[i]))
        pDau2 = (1.0,pDauphi2[i],u.pseudoRapToPolar(pDaueta2[i]))
        alpha = u.angleBetweenVectorsInPolarCoordinats(pDau1,pDau2)
        dauE2 = fun.bEiofbEj(H_mass[i],l[1],alpha)
        daue2array.append(dauE2)
        fi.write(str(l[1])+'\t'+str(dauE2)+'\t'+str(dauE2+l[1])+'\n')
        

    
    
    fi.close()


np.save('bjet2_e_fitted',daue2array)
np.save('dau2_e_fitted',be2array)
p.plot1d2(be2array,'./','B_e2 fitted',xlim=[0,1000])
p.plot1d2(daue2array,'./','Dau_e2 fitted',xlim=[0,1000])
