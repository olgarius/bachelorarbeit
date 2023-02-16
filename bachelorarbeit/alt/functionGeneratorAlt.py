import numpy as np

import util as u
import functions as f


class functionGenerator:
    def __init__(self,events,sigma_be,sigma_daue):
        self.data = events
        self.fs = []
        self.fdimss = []
        self.genBBETerm(sigma_be)
        self.genDauDauETerm(sigma_daue)
        
  
    def genBBETerm(self,sigma_be):
        H_mass = u.structToArray(self.data,'bH_mass')
        B1_e = u.structToArray(self.data,'bjet1_e') 
        pBeta1 = u.structToArray(self.data,'bjet1_eta')
        pBeta2 = u.structToArray(self.data,'bjet2_eta')
        pBphi1 = u.structToArray(self.data,'bjet1_phi')
        pBphi2 = u.structToArray(self.data,'bjet2_phi')

        
        

        farray = []

        for i,_ in enumerate(H_mass):
            pB1 = (1.0,pBphi1[i],u.pseudoRapToPolar(pBeta1[i]))
            pB2 = (1.0,pBphi2[i],u.pseudoRapToPolar(pBeta2[i]))
            alpha = u.angleBetweenVectorsInPolarCoordinats(pB1,pB2)
            
            bE2 = f.bEiofbEj(H_mass[i],B1_e[i],alpha)
                
            farray.append(f.makeChi2(bE2,sigma_be))

        self.fs.append(farray)
        self.fdimss.append([1])
    
    def genDauDauETerm(self,sigma_daue):
        H_mass = u.structToArray(self.data,'tauH_mass')
        dau1_e = u.structToArray(self.data,'dau1_e') 
        pDaueta1 = u.structToArray(self.data,'dau1_eta')
        pDaueta2 = u.structToArray(self.data,'dau2_eta')
        pDauphi1 = u.structToArray(self.data,'dau1_phi')
        pDauphi2 = u.structToArray(self.data,'dau2_phi')

        farray = []

        for i,_ in enumerate(H_mass):
            pDau1 = (1.0,pDauphi1[i],u.pseudoRapToPolar(pDaueta1[i]))
            pDau2 = (1.0,pDauphi2[i],u.pseudoRapToPolar(pDaueta2[i]))
            alpha = u.angleBetweenVectorsInPolarCoordinats(pDau1,pDau2)
            
            dauE2 = f.dauEiofdauEj(H_mass[i],dau1_e[i],alpha)
                
            farray.append(f.makeChi2(dauE2,sigma_daue))

        self.fs.append(farray)
        self.fdimss.append([2])



    def getFunctionBatch(self):
        return np.array(self.fs).transpose().tolist()