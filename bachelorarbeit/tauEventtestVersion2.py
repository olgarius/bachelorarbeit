import grid

import structutil as su
import datautil as du
import mathutil as mu
import plotting as plo

import matplotlib.pyplot as plt
import json


import numpy as np
import scipy.optimize as sopt


from PATHS import WORKINGDATAPATH

import time

EPSILON = 0.9
EPSILON2 = 20

TAUERR = 0.1

MHIGGS = 125.25


#load data and select relevant data
events = np.load(WORKINGDATAPATH)
# events = data['events']

unitAxis = [np.linspace(0,1,100),np.linspace(0,1,100)]



chi2 = []
gridarr = []
chi2str =[]
mins = []
sigmaFits = []
minpos = []

tau1_ex = su.structToArray(events,'dau1_ex')
tau1_ey = su.structToArray(events,'dau1_ey')
tau1_ez = su.structToArray(events,'dau1_ez')

tau2_ex = su.structToArray(events,'dau2_ex')
tau2_ey = su.structToArray(events,'dau2_ey')
tau2_ez = su.structToArray(events,'dau2_ez')

met_x = su.structToArray(events,'met_x')
met_y = su.structToArray(events,'met_y')




TauE1_gen = su.structToArray(events,'genLepton1_e') 
TauE2_gen = su.structToArray(events,'genLepton2_e') 


TauE1_mes = su.structToArray(events,'dau1_e') 
TauE2_mes = su.structToArray(events,'dau2_e') 
# TauE1_eta = su.structToArray(events,'dau1_eta') 
# TauE2_eta = su.structToArray(events,'dau2_eta') 
# TauE1_phi = su.structToArray(events,'dau1_phi') 
# TauE2_phi = su.structToArray(events,'dau2_phi')

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

for i in range(10):
   
    start= time.perf_counter()

    def fitfunc(x):
        return fac[i]/x

    # evaluationAxisTau1E = np.linspace(EPSILON*TauE1_mes[i],MHIGGS**2/TauHM_mes[i]**2 * 1/EPSILON*TauE1_mes[i],100)
    evaluationAxisTau1E = mu.axisValueTransformToMu1Mu2(unitAxis[0],TauE1_mes[i], TauE2_mes[i], TAUERR*TauE1_mes[i], TAUERR*TauE2_mes[i],fitfunc,3)
    # evaluationAxisTau1E = np.linspace(TauE1_mes[i]-EPSILON2,MHIGGS**2/TauHM_mes[i]**2 * TauE1_mes[i]*TauE2_mes[i] /(TauE2_mes[i]-EPSILON2),100)
    # evaluationAxisTau1E = np.linspace(TauE1_mes[i],fac[i]/(0.99*TauE2_mes[i]),100)
    # print(evaluationAxisTau1E)

    evaluationAxisf = np.linspace(0,1,101)


    evaluationAxisArr.append(evaluationAxisTau1E)
    
    def f1chi2(x,f1):
        tau1_e = np.sqrt((tau1_ex[i]+f1*met_x[i])**2+(tau1_ey[i]+f1*met_y[i])**2+tau1_ez[i]**2)
        return ((tau1_e-x)/(TAUERR*TauE1_mes[i]))**2


    def f2chi2(x,f1):
        tau2_e = np.sqrt((tau2_ex[i]+(1-f1)*met_x[i])**2+(tau2_ey[i]+(1-f1)*met_y[i])**2+tau2_ez[i]**2)
        return ((tau2_e-fac[i]/x)/TauE2_mes[i]*TAUERR)**2
        
   


    g = grid.grid()
    g.addDimension(evaluationAxisTau1E)
    g.addDimension(evaluationAxisf)
    

    g.addFunction(f1chi2, [1,2])
    g.addFunction(f2chi2, [1,2])
    


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
    chi2str.append(json.dumps(chi2s.tolist()))
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
    for j in reversed(range(1,g.dimension)):
        tempGrid = np.amin(tempGrid,j)
    tempGrid2 = tempGrid
    


    

    tempGrid = tempGrid - minvalue - 1
    # print(tempGrid)

    if tempGrid[0] <0 or tempGrid[-1] <0:
        # print('sigma OOB ', tempGrid[-0], tempGrid[-1]) 
        sigmaFit = -1
    else:
        # print(mp[0][0])
        tempGrid = tempGrid**2
        gA = tempGrid[0:mp[0][0]]
        gB = tempGrid[mp[0][0]:(len(tempGrid)-1)]
        left = evaluationAxisTau1E[np.where(min(gA)==gA)[0][0]]
        right = evaluationAxisTau1E[np.where(min(gB)==gB)[0][-1]+len(gA)]
        sigmaFit = right - left
        if sigmaFit < 0:
            print(tempGrid)

        print(left,right, sigmaFit,i)
    sigmaFits.append(sigmaFit)

    print('Time Elapsed: ', time.perf_counter()-start)

    xdelta = evaluationAxisTau1E[99]-evaluationAxisTau1E[0]
    xlim = [evaluationAxisTau1E[0] -0.1*xdelta,evaluationAxisTau1E[99] + 0.1*xdelta]

    ydelta = max(tempGrid2)-min(tempGrid2)
    ylim= [min(tempGrid2)-0.1*ydelta,max(tempGrid2)+0.1*ydelta]


    plo.scatter('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/plots/tauTestPlots2/', r"$\chi^2 \tau$-Fit Event "+str(i),r"$E_\tau^f$ in GeV",r"$\chi^2$-Value",(evaluationAxisTau1E,tempGrid2,'Event'+str(i)),(evaluationAxisTau1E[np.where(np.amin(tempGrid2)==tempGrid2)[0]][0],min(tempGrid2),r'$E_{\tau 1}^f$'), (TauE1_mes[i],min(tempGrid2),r'$E_{\tau 1}^m$'), (TauE1_gen[i],min(tempGrid2),r'$E_{\tau 1}^g$'),(fac[i]/TauE2_mes[i],min(tempGrid2),r'$E_{\tau 1}(E_{\tau 2}^m$)'),(fac[i]/TauE2_gen[i],min(tempGrid2),r'$E_{\tau 1}(E_{\tau 2}^g$)'),lim=xlim,yLim=ylim,alttitle='Chi2TauEvent'+str(i),s=8)