import plotting
import matplotlib.pyplot as plt
import numpy as np
import json
import datautil as du

import plotting as plo

from PATHS import WORKINGDATAPATH

data = np.load(WORKINGDATAPATH)

# axis = data['ax1Values']
# chi2vals = data['chi2']
TauE1_mes = data['dau1_e']
TauE2_mes = data['dau2_e']

TauE1_gen = data['genLepton1_e']
TauE2_gen = data['genLepton2_e']



fac = data['tauie_to_tauje']
be1be2measpos = fac/TauE2_mes
minpos = data['dau_chi2val']
be1fit = data['fitdau1_e']
be2fit = data['fitdau2_e']
be1be2fitpos = fac/be2fit

dicts = data['dau_chi2dict']


invCovMat = du.combineArrayToMatrixArray(data['met_invcov00'], data['met_invcov01'], data['met_invcov01'], data['met_invcov11'])

indices = []
xAxis = []
values = []

for ds in dicts:
    d = json.loads(ds)
    indices.append(d['index'])
    xAxis.append(d['xAxis'])
    values.append(d['values'])

edgeProblem = np.where(data['taufit_valueOnEdge'])[0][0:10]


for i in edgeProblem:
    fig = plt.figure()

    print(invCovMat[i])

    ax = np.array(xAxis[i])*TauE1_mes[i]
    ay = values[i]

    # plt.scatter(ax ,ay, marker='x', label = 'chi2')
    # plt.axvline(TauE1_mes[i], c='red', label = 'E_tau1 measured ' + str(TauE1_mes[i]))
    # plt.axvline(be1be2measpos[i], c='orange', label = 'E_tau2 measured mapped to Etau1 '+ str(be1be2measpos[i]))
    # plt.axvline(be1fit[i], c='magenta', label = 'Eb1 fit')
    # plt.legend(fontsize=10)
    # plt.xlabel('E_tau1')
    # plt.ylabel('Chi2')


    # plt.savefig('chi2forsingleevent_tau' + str(i) + '.png')

    xdelta = ax[99]-ax[0]
    xlim = [ax[0] -0.1*xdelta,ax[99] + 0.1*xdelta]

    ydelta = max(ay)-min(ay)
    ylim= [min(ay)-0.1*ydelta,max(ay)+0.1*ydelta]

    plo.scatter('', r"$\chi^2 \tau$-Fit Event "+str(i),r"$E_\tau^f$ in GeV",r"$\chi^2$-Value",(ax,ay,'Event'+str(i)),(ax[np.where(np.amin(ay)==ay)[0]][0],min(ay),r'$E_{\tau 1}^f$'), (TauE1_mes[i],min(ay),r'$E_{\tau 1}^m$'), (TauE1_gen[i],min(ay),r'$E_{\tau 1}^g$'),(fac[i]/TauE2_mes[i],min(ay),r'$E_{\tau 1}(E_{\tau 2}^m$)'),(fac[i]/TauE2_gen[i],min(ay),r'$E_{\tau 1}(E_{\tau 2}^g$)'),lim=xlim,yLim=ylim,alttitle='Chi2TauEvent'+str(i),s=8)

    


