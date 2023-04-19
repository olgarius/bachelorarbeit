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


#load data and select relevant data
events = np.load(WORKINGDATAPATH)
# events = data['events']

unitAxis = [np.linspace(0,1,100),np.linspace(0,1,100)]



chi2 = []
chi2str =[]
mins = []
sigmaFits = []
minpos = []

# tau1_ex = su.structToArray(events,'dau1_ex')
# tau1_ey = su.structToArray(events,'dau1_ey')
# tau1_ez = su.structToArray(events,'dau1_ez')

# tau2_ex = su.structToArray(events,'dau2_ex')
# tau2_ey = su.structToArray(events,'dau2_ey')
# tau2_ez = su.structToArray(events,'dau2_ez')

# met_x = su.structToArray(events,'met_x')
# met_y = su.structToArray(events,'met_y')

# TauE1_mes = su.structToArray(events,'dau1_e') 
# TauE2_mes = su.structToArray(events,'dau2_e') 

# fac = su.structToArray(events,'tauie_to_tauje')

# Debug:

tau1_ex = su.structToArray(events,'debugdau1_ex')
tau1_ey = su.structToArray(events,'debugdau1_ey')
tau1_ez = su.structToArray(events,'debugdau1_ez')

tau2_ex = su.structToArray(events,'debugdau2_ex')
tau2_ey = su.structToArray(events,'debugdau2_ey')
tau2_ez = su.structToArray(events,'debugdau2_ez')

met_x = su.structToArray(events,'debugmet_x')
met_y = su.structToArray(events,'debugmet_y')

TauE1_mes = su.structToArray(events,'debugdau1_e') 
TauE2_mes = su.structToArray(events,'debugdau2_e') 

fac = su.structToArray(events,'gentauie_to_gentauje')

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



minvalues = [] 
highminvalue =[]
highminvalueAxis =[]
highminvalueIndex = []

chi2dicts = [] 

for i, n in enumerate(TauE1_mes):

    def fitfunc(x):
        return fac[i]/x


    evaluationAxisTau1E = mu.axisValueTransformToMu1Mu2(unitAxis[0],TauE1_mes[i], TauE2_mes[i], TAUERR*TauE1_mes[i], TAUERR*TauE2_mes[i],fitfunc,3)
    # evaluationAxisTau1E = np.linspace(EPSILON*n,MHIGGS**2/TauHM_mes[i]**2 * 1/EPSILON*n,100)
    # evaluationAxisTau1E = np.linspace(n-EPSILON2,MHIGGS**2/TauHM_mes[i]**2 * n*TauE2_mes[i] /(TauE2_mes[i]-EPSILON2),100)
    # evaluationAxisTau1E = np.linspace(TauE1_mes[i],fac[i]/(0.99*TauE2_mes[i]),100)
    # print(evaluationAxisTau1E)

    evaluationAxisf = np.linspace(0,1,100)


    evaluationAxisArr.append(evaluationAxisTau1E)
    

    def f1chi2(x,f1):
        tau1_e = np.sqrt((tau1_ex[i]+f1*met_x[i])**2+(tau1_ey[i]+f1*met_y[i])**2+tau1_ez[i]**2,dtype=np.float32)
        return ((tau1_e-x)/(TAUERR*TauE1_mes[i]))**2 


    def f2chi2(x,f1):
        tau2_e = np.sqrt((tau2_ex[i]+(1-f1)*met_x[i])**2+(tau2_ey[i]+(1-f1)*met_y[i])**2+tau2_ez[i]**2,dtype=np.float32)
        return ((tau2_e-fac[i]/x)/(TauE2_mes[i]*TAUERR))**2 


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
    g.addDimension(evaluationAxisTau1E)
    g.addDimension(evaluationAxisf)
    # g.addDimension(evaluationAxisf)

    g.addFunction(f1chi2, [1,2])
    g.addFunction(f2chi2, [1,2])

    # g.addFunction(f1chi2, [1,2])
    # g.addFunction(f2chi2, [1,3])
    # g.addFunction(f3chi2, [2,3])


    g.evaluate()
    m = g.getMinCoords()
    mp = g.getMin()
    minpos.append(mp)


    if min(mp[0]) == 0 or max(mp[0]) == 99:
        counterEdge += 1
        edgeProblem.append(g.evaluated)
        edgeProblemAxis.append(evaluationAxisTau1E)
        edgeProblemIndex.append(i)


    chi2.append(g.evaluated)
    chi2s = g.evaluated
    # chi2str.append(json.dumps(chi2s.tolist()))
    mins.append(m[0])

    minvalue = np.min(g.evaluated)
    minvalues.append(minvalue)
    if minvalue > 10:
        highminvalue.append(g.evaluated)
        highminvalueAxis.append(evaluationAxisTau1E)
        highminvalueIndex.append(i)

    # def f(x,y,z):
    #     return f1chi2(x,y)+f2chi2(x,z)+f3chi2(y,z) -minvalue-1
    # root1 = sopt.fsolve(f,evaluationAxisTau1E[0])
    # root2 = sopt.fsolve(f,evaluationAxisTau1E[99])


    # sigmaFit = root2[0]  -root1[0] 

    tempGrid = g.evaluated
    for j in reversed(range(g.dimension-1)):
        tempGrid = np.amin(tempGrid,j)
    
    tempGrid = tempGrid - minvalue - 1
    # print(tempGrid)

    if tempGrid[0] <0 or tempGrid[-1] <0:
        # print('sigma OOB ', tempGrid[-0], tempGrid[-1]) 
        sigmaFit = -1
        sc1+=1
    else:
        # print(mp[0][0])
        tempGrid = tempGrid**2
        gA = tempGrid[0:mp[0][0]]
        gB = tempGrid[mp[0][0]:(len(tempGrid)-1)]
        if len(gB) is 0 or len(gA) is 0:
            sc2+=1
            sigmaFit = -1
        else:
            left = evaluationAxisTau1E[np.where(min(gA)==gA)[0][0]]
            right = evaluationAxisTau1E[np.where(min(gB)==gB)[0][-1]+len(gA)]
            sigmaFit = right - left
       
    sigmaFits.append(sigmaFit)

    # if sigmaFit < 0:
    #     counterNegSigma +=1

    # sigmaFits.append(sigmaFit)

    chi2dict = {'index' : indices[i], 'xAxis' :  evaluationAxisTau1E.tolist(), 'values' : g.evaluated.tolist()}
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
su.updateDataSet(WORKINGDATAPATH,'fitdau1_e', mins)
su.updateDataSet(WORKINGDATAPATH,'fitdau2_e',fac/mins)
su.updateDataSet(WORKINGDATAPATH,'dau_chi2val', minvalues)
su.updateDataSet(WORKINGDATAPATH,'fitdau1_esigma',sigmaFits)
# su.updateDataSet(WORKINGDATAPATH, 'dau_chi2dict', chi2dicts) To memory intense


# print('EP indices: ',edgeProblemIndex)



if len(edgeProblemIndex) > 1:
    fig = plt.figure()

    plt.scatter(evaluationAxisArr[edgeProblemIndex[0]],np.amin(chi2[edgeProblemIndex[0]],1))



    plt.savefig('./TESTEPTau.png')

    fig = plt.figure()

    plt.scatter(evaluationAxisArr[edgeProblemIndex[1]],np.amin(chi2[edgeProblemIndex[1]],1))
    plt.savefig('./TESTEP1Tau.png')

fig = plt.figure()




plt.scatter(evaluationAxisArr[0],np.amin(chi2[0],1))
plt.savefig('./TEST2Tau.png')

fig = plt.figure()

if len(highminvalueIndex) > 1:
    plt.scatter(evaluationAxisArr[highminvalueIndex[0]],np.amin(chi2[highminvalueIndex[0]],1))
    plt.savefig('./TESTHM1Tau.png')