import grid

import structutil as su
import datautil as du
import mathutil as mu

import matplotlib.pyplot as plt
import json


import numpy as np
import scipy.optimize as sopt


from PATHS import WORKINGDATAPATH

EPSILON = 0.7
EPSILON2 = 20

TAUERR = 0.1

MHIGGS = 125.25

amin = 1

#load data and select relevant data
events = np.load(WORKINGDATAPATH)
# events = data['events']

unitAxis = [np.linspace(0,1,100),np.linspace(0,1,100)]



chi2 = []
chi2str =[]
mins = []
sigmaFits = []
minpos = []
newMet_x = []
newMet_y = []


tau1_ex = su.structToArray(events,'dau1_ex')
tau1_ey = su.structToArray(events,'dau1_ey')
tau1_ez = su.structToArray(events,'dau1_ez')

tau2_ex = su.structToArray(events,'dau2_ex')
tau2_ey = su.structToArray(events,'dau2_ey')
tau2_ez = su.structToArray(events,'dau2_ez')

met_x = su.structToArray(events,'met_x')
met_y = su.structToArray(events,'met_y')

TauE1_mes = su.structToArray(events,'dau1_e') 
TauE2_mes = su.structToArray(events,'dau2_e') 

fac = su.structToArray(events,'tauie_to_tauje')

# Debug:

# tau1_ex = su.structToArray(events,'debugdau1_ex')
# tau1_ey = su.structToArray(events,'debugdau1_ey')
# tau1_ez = su.structToArray(events,'debugdau1_ez')

# tau2_ex = su.structToArray(events,'debugdau2_ex')
# tau2_ey = su.structToArray(events,'debugdau2_ey')
# tau2_ez = su.structToArray(events,'debugdau2_ez')

# met_x = su.structToArray(events,'debugmet_x')
# met_y = su.structToArray(events,'debugmet_y')

# TauE1_mes = su.structToArray(events,'debugdau1_e') 
# TauE2_mes = su.structToArray(events,'debugdau2_e') 

# fac = su.structToArray(events,'gentauie_to_gentauje')

sc1= 0
sc2 =0


# TauE1_eta = su.structToArray(events,'dau1_eta') 
# TauE2_eta = su.structToArray(events,'dau2_eta') 
# TauE1_phi = su.structToArray(events,'dau1_phi') 
# TauE2_phi = su.structToArray(events,'dau2_phi')

TauHM_mes = su.structToArray(events,'tauH_mass')

indices = su.structToArray(events, 'index')
invCovMat = du.combineArrayToMatrixArray(events['met_invcov00'], events['met_invcov01'], events['met_invcov01'], events['met_invcov11'])

evaluationAxisArr = []


counterNegSigma = 0

counterEdge = 0 
edgeProblem = []
edgeProblemAxis = [] 
edgeProblemIndex = []
edgeProblemBool = []


minvalues = [] 
highminvalue =[]
highminvalueAxis =[]
highminvalueIndex = []

chi2dicts = [] 

for i, n in enumerate(TauE1_mes):

    
    evaluationAxis = np.linspace(amin,fac[i]/(TauE1_mes[i]*TauE2_mes[i]*amin),100)
  


    evaluationAxisArr.append(evaluationAxis)
    
    def f1chi2(a):

        pTmiss = np.c_[met_x[i],met_y[i]]
        


        pTt1 = np.c_[tau1_ex[i],tau1_ey[i]]
        pTt2 = np.c_[tau2_ex[i],tau2_ey[i]]

        b = fac[i]/(a*TauE1_mes[i]*TauE2_mes[i])

        delta = -pTmiss + (a-1)[:,np.newaxis]*pTt1 +(b-1)[:,np.newaxis] * pTt2

        


        return np.einsum('ij,ij->i',delta, np.einsum('ji,lj->li',invCovMat[i],delta))
    
    def f1chi2_nonvector(a):

        pTmiss = np.c_[met_x[i],met_y[i]]
        


        pTt1 = np.c_[tau1_ex[i],tau1_ey[i]]
        pTt2 = np.c_[tau2_ex[i],tau2_ey[i]]

        b = fac[i]/(a*TauE1_mes[i]*TauE2_mes[i])

        delta = -pTmiss + (a-1)*pTt1 +(b-1) * pTt2
      
        d1 = np.dot(invCovMat[i],delta[0])

        return np.dot(delta,d1)



    # def f1chi2(x,f1):
    #     tau1_e = np.sqrt((tau1_ex[i]+f1*met_x[i])**2+(tau1_ey[i]+f1*met_y[i])**2+tau1_ez[i]**2)
    #     return ((tau1_e-x)/(0.03*n))**2


    # def f2chi2(x,f2):
    #     tau2_e = np.sqrt((tau2_ex[i]+f2*met_x[i])**2+(tau2_ey[i]+f2*met_y[i])**2+tau2_ez[i]**2)
    #     return ((tau2_e-fac[i]/x)/TauE2_mes[i]*0.03)**2
        
    # def f3chi2(f1,f2):
        
    #     f12 = 1-(np.add(f1,f2))
    #     met = np.c_[met_x[i],met_y[i]]
    #     delta = met * f12[:,:,np.newaxis]

    #     return np.einsum('ilj,ilj->il',delta, np.einsum('ji,klj->kli',invCovMat[i],delta))

    # def f3chi2(f1,f2):
    #     return (1-f1-f2)**2


    g = grid.grid()
    g.addDimension(evaluationAxis)
    

    g.addFunction(f1chi2, [1])
    

    


    g.evaluate()
    m = g.getMinCoords()
    mp = g.getMin()
    minpos.append(mp)


    if min(mp[0]) == 0 or max(mp[0]) == 99:
        counterEdge += 1
        edgeProblem.append(g.evaluated)
        edgeProblemAxis.append(evaluationAxis)
        edgeProblemIndex.append(i)
        edgeProblemBool.append(True)
    else:
        edgeProblemBool.append(False)

    a = m[0]
    b= fac[i]/(a*TauE1_mes[i]*TauE2_mes[i])

    newMet_x.append(met_x[i] - (a-1)*tau1_ex[i] - (b-1)*tau2_ex[i])
    newMet_y.append(met_y[i] - (a-1)*tau1_ey[i] - (b-1)*tau2_ey[i])



    chi2.append(g.evaluated)
    chi2s = g.evaluated
    # chi2str.append(json.dumps(chi2s.tolist()))
    mins.append(m[0])

    minvalue = np.min(g.evaluated)
    minvalues.append(minvalue)
    if minvalue > 10:
        highminvalue.append(g.evaluated)
        highminvalueAxis.append(evaluationAxis)
        highminvalueIndex.append(i)

    # def f(x):
    #     return f1chi2(x) -minvalue-1
    # root1 = sopt.fsolve(f,evaluationAxis[0])
    # root2 = sopt.fsolve(f,evaluationAxis[99])
    # sigmaFit = root2[0]  -root1[0] 

    

    def f(x):
        return f1chi2_nonvector(x) -minvalue-1

    maxi = 2*evaluationAxis[mp[0]]
    while f(maxi) < 0:
        maxi = maxi*2
    mini = 0.5*evaluationAxis[mp[0]]
    while f(mini) < 0:
        mini = mini/2

    root1 = sopt.brentq(f,mini,evaluationAxis[mp[0]])
    root2 = sopt.brentq(f,evaluationAxis[mp[0]],maxi)

    # print(i, root1, root2, mp[0], f(evaluationAxis[mp[0]]))

    sigmaFit = (root2  -root1)/2

           
    sigmaFits.append(sigmaFit)

    if sigmaFit < 0:
        counterNegSigma +=1

    chi2dict = {'index' : indices[i], 'xAxis' :  evaluationAxis.tolist(), 'values' : g.evaluated.tolist()}
    chi2dicts.append(json.dumps(chi2dict))
    
    if i%1000 == 0:
        print(i)

print('Negative sigma percentage: ',counterNegSigma/len(TauE1_mes)*100, '%')
print('Out of bounds min percentage: ',counterEdge/len(TauE2_mes)*100, '%')
print('Chi2 > 3.84 percentage: ',len(np.where(np.array(minvalues) > 3.84)[0])/len(minvalues)*100, '%')

print(np.average(minpos),np.std(minpos))
# print(min(minpos),max(minpos))

print(sc1,sc2)

# su.updateDataSetWithFloatArray(WORKINGDATAPATH,'chi2',chi2)
# su.updateDataSetWithFloatArray(WORKINGDATAPATH,'ax1Values',evaluationAxisArr)
su.updateDataSet(WORKINGDATAPATH,'fitdau1_e', mins*TauE1_mes)
su.updateDataSet(WORKINGDATAPATH,'fitdau_a', mins)

su.updateDataSet(WORKINGDATAPATH,'fitdau2_e',fac/(mins*TauE1_mes))
su.updateDataSet(WORKINGDATAPATH,'dau_chi2val', minvalues)
su.updateDataSet(WORKINGDATAPATH,'fitdau1_esigma',sigmaFits*TauE1_mes)
su.updateDataSet(WORKINGDATAPATH,'fitdau_asigma',sigmaFits)

su.updateDataSet(WORKINGDATAPATH,'fitmet_x',newMet_x)
su.updateDataSet(WORKINGDATAPATH,'fitmet_y',newMet_y)
su.updateDataSet(WORKINGDATAPATH,'taufit_valueOnEdge',edgeProblemBool)


su.updateDataSet(WORKINGDATAPATH, 'dau_chi2dict', chi2dicts) 


# print('EP indices: ',edgeProblemIndex)



if len(edgeProblemIndex) > 1:
    fig = plt.figure()

    plt.scatter(evaluationAxisArr[edgeProblemIndex[0]],chi2[edgeProblemIndex[0]])



    plt.savefig('./TESTEPTau.png')

    fig = plt.figure()

    plt.scatter(evaluationAxisArr[edgeProblemIndex[1]],chi2[edgeProblemIndex[1]])
    plt.savefig('./TESTEP1Tau.png')

fig = plt.figure()




plt.scatter(evaluationAxisArr[0],chi2[0])
plt.savefig('./TEST2Tau.png')

fig = plt.figure()

if len(highminvalueIndex) > 1:
    plt.scatter(evaluationAxisArr[highminvalueIndex[0]],chi2[highminvalueIndex[0]])
    plt.savefig('./TESTHM1Tau.png')