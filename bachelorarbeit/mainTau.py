import grid

import structutil as su
import datautil as du
import mathutil as mu

import matplotlib.pyplot as plt
import json


import numpy as np
import scipy.optimize as sopt


from PATHS import WORKINGDATAPATH

EPSILON = 0.9
MHIGGS = 125.25


#load data and select relevant data
events = np.load(WORKINGDATAPATH)
# events = data['events']

unitAxis = [np.linspace(0,1,100),np.linspace(0,1,100)]



chi2 = []
chi2str =[]
mins = []
sigmaFits = []
minpos = []

TauE1_mes = su.structToArray(events,'dau1_e') 
TauE2_mes = su.structToArray(events,'dau2_e') 
TauE1_eta = su.structToArray(events,'dau1_eta') 
TauE2_eta = su.structToArray(events,'dau2_eta') 
TauE1_phi = su.structToArray(events,'dau1_phi') 
TauE2_phi = su.structToArray(events,'dau2_phi')

TauHM_mes = su.structToArray(events,'tauH_mass')
fac = su.structToArray(events,'tauie_to_tauje')
indices = su.structToArray(events, 'index')
invCovMat = du.combineArrayToMatrixArray(events['met_invcov00'], events['met_invcov01'], events['met_invcov01'], events['met_invcov11'])

evaluationAxisArr = []


counterNegSigma = 0

counterEdge = 0 
edgeProblem = []
edgeProblemAxis = [] 
edgeProblemIndex = []



minvalues = [] 
highminvalue =[]
highminvalueAxis =[]
highminvalueIndex = []

chi2dicts = [] 

for i, n in enumerate(TauE1_mes):



    
    
    

    evaluationAxis = np.linspace(EPSILON*n,MHIGGS**2/TauHM_mes[i]**2 * 1/EPSILON*n,100)
    evaluationAxisArr.append(evaluationAxis)
    

    def f1chi2(x):
        return ((x-n)/(0.1*n))**2
    
    def f2chi2(x):
        return (((fac[i]/x)-TauE2_mes[i])/(TauE2_mes[i]*0.1))**2

    def f3chi2(x):
        thetaTau1 = mu.pseudoRapToPolar(TauE1_eta[i])
        thetaTau2 = mu.pseudoRapToPolar(TauE2_eta[i])
        Tau1scalar = (n-x)*np.sin(thetaTau1)
        Tau1Vector = np.c_[np.cos(TauE1_phi[i]),np.sin(TauE1_phi[i])] * Tau1scalar[:,np.newaxis]

        Tau2scalar = (TauE2_mes[i]-fac[i]/x)*np.sin(thetaTau2)
        Tau2Vector = np.c_[np.cos(TauE2_phi[i]),np.sin(TauE2_phi[i])] * Tau2scalar[:,np.newaxis]


        delta =  Tau1Vector +  Tau2Vector

        # return np.matmul(delta, np.matmul(invCov,delta))
        return np.einsum('ij,ij->i',delta, np.einsum('ji,kj->ki',invCovMat[i],delta))

    g = grid.grid()
    g.addDimension(evaluationAxis)
    g.addFunction(f1chi2, [1])
    g.addFunction(f2chi2, [1])
    g.addFunction(f3chi2, [1])


    g.evaluate()
    m = g.getMinCoords()
    mp = g.getMin()
    minpos.append(mp)


    if min(mp[0]) == 0 or max(mp[0]) == 99:
        counterEdge += 1
        edgeProblem.append(g.evaluated)
        edgeProblemAxis.append(evaluationAxis)
        edgeProblemIndex.append(i)


    chi2.append(g.evaluated)
    chi2s = g.evaluated
    chi2str.append(json.dumps(chi2s.tolist()))
    mins.append(m[0])

    minvalue = np.min(g.evaluated)
    minvalues.append(minvalue)
    if minvalue > 10:
        highminvalue.append(g.evaluated)
        highminvalueAxis.append(evaluationAxis)
        highminvalueIndex.append(i)

    def f(x):
        return f1chi2(x)+f2chi2(x)+f3chi2(x) -minvalue-1
    root1 = sopt.fsolve(f,evaluationAxis[0])
    root2 = sopt.fsolve(f,evaluationAxis[99])


    sigmaFit = root2[0]  -root1[0] 

    if sigmaFit < 0:
        counterNegSigma +=1

    sigmaFits.append(sigmaFit)

    chi2dict = {'index' : indices[i], 'xAxis' :  evaluationAxis.tolist(), 'values' : g.evaluated.tolist()}
    chi2dicts.append(json.dumps(chi2dict))

print('Negative sigma percentage: ',counterNegSigma/len(TauE1_mes)*100, '%')
print('Out of bounds min percentage: ',counterEdge/len(TauE2_mes)*100, '%')
print('Chi2 > 3.84 percentage: ',len(np.where(np.array(minvalues) > 3.84))/len(minvalues)*100, '%')

print(np.average(minpos),np.std(minpos))
# print(min(minpos),max(minpos))



# su.updateDataSetWithFloatArray(WORKINGDATAPATH,'chi2',chi2)
# su.updateDataSetWithFloatArray(WORKINGDATAPATH,'ax1Values',evaluationAxisArr)
su.updateDataSet(WORKINGDATAPATH,'fitdau1_e', mins)
su.updateDataSet(WORKINGDATAPATH,'fitdau2_e',fac/mins)
su.updateDataSet(WORKINGDATAPATH,'dau_chi2val', minvalues)
su.updateDataSet(WORKINGDATAPATH,'fitdau1_esigma',sigmaFits)
su.updateDataSet(WORKINGDATAPATH, 'dau_chi2dict', chi2dicts)


print('EP indices: ',edgeProblemIndex)



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