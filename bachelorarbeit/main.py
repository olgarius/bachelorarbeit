import grid
import functionGenerator
import util 
import functions
import plotting

import matplotlib.pyplot as plt


import numpy as np
import scipy.optimize as sopt

from pathlib import Path

DATAPATH = Path('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet.npy')



#load data and select relevant data
events = np.load(DATAPATH)
# events = data['events']

unitAxis = [np.linspace(0,1,100),np.linspace(0,1,100)]

be1functions, dbe1 = functionGenerator.getBBE1Chi2(events)
be2functions, dbe2, be2fitfunctions = functionGenerator.getBBE2Chi2(events)


chi2 = []
chi2str =[]
mins = []
sigmaFits = []
minpos = []

BE1_mes = util.structToArray(events,'bjet1_e') 
BE2_mes = util.structToArray(events,'bjet2_e') 
BE1sigma = util.structToArray(events, 'bjet1_sigma')
BE2sigma = util.structToArray(events, 'bjet2_sigma')
fac = util.structToArray(events,'bie_to_bje')

sigmaFits = []
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

for i, n in enumerate(be1functions):



    # evaluationAxis = util.axisValueTransformToMu(unitAxis[0],BE1_mes[i],BE1sigma[i],6)
    evaluationAxis = util.axisValueTransformToMu1Mu2(unitAxis[0],BE1_mes[i], BE2_mes[i], BE1sigma[i], BE1sigma[i],be2fitfunctions[i],3)
    # evaluationAxis = unitAxis[0]*300
    evaluationAxisArr.append(evaluationAxis)

    g = grid.grid()
    g.addDimension(evaluationAxis)
    g.addFunction(n, dbe1)
    g.addFunction(be2functions[i], dbe2)

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
    chi2str.append(str(g.evaluated))
    mins.append(m[0])

    minvalue = np.min(g.evaluated)
    minvalues.append(minvalue)
    if minvalue > 10:
        highminvalue.append(g.evaluated)
        highminvalueAxis.append(evaluationAxis)
        highminvalueIndex.append(i)

    def f(x):
        return n(x)+be2functions[i](x)-minvalue-1
    root1 = sopt.fsolve(f,evaluationAxis[0])
    root2 = sopt.fsolve(f,evaluationAxis[99])


    sigmaFit = root2[0]-root1[0] 

    if sigmaFit < 0:
        counterNegSigma +=1

    sigmaFits.append(sigmaFit)

print('Negative sigma percentage: ',counterNegSigma/len(be1functions)*100, '%')
print('Out of bounds min percentage: ',counterEdge/len(be1functions)*100, '%')
print(np.average(minpos),np.std(minpos))
# print(min(minpos),max(minpos))

print(chi2[0].shape)

util.updateDataSet(DATAPATH,'chi2',chi2str)
util.updateDataSet(DATAPATH,'fitbjet1_e', mins)
util.updateDataSet(DATAPATH,'fitbjet2_e',fac/mins)
util.updateDataSet(DATAPATH,'chi2val', minvalues)
util.updateDataSet(DATAPATH,'fitbjet1_esigma',sigmaFits)



fig = plt.figure()

plt.scatter(evaluationAxisArr[edgeProblemIndex[0]],chi2[edgeProblemIndex[0]])



plt.savefig('./TESTEP.png')

fig = plt.figure()

plt.scatter(evaluationAxisArr[edgeProblemIndex[1]],chi2[edgeProblemIndex[1]])
plt.savefig('./TESTEP1.png')

fig = plt.figure()

plt.scatter(evaluationAxisArr[0],chi2[0])
plt.savefig('./TEST2.png')

fig = plt.figure()

plt.scatter(evaluationAxisArr[highminvalueIndex[0]],chi2[highminvalueIndex[0]])
plt.savefig('./TESTHM1.png')