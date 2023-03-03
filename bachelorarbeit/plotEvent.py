import plotting
import matplotlib.pyplot as plt
import numpy as np
import json


from PATHS import WORKINGDATAPATH

data = np.load(WORKINGDATAPATH)

# axis = data['ax1Values']
# chi2vals = data['chi2']
be1meas = data['bjet1_e']
be2meas = data['bjet2_e']
fac = data['bie_to_bje']
be1be2measpos = fac/be2meas
minpos = data['chi2val']
be1fit = data['fitbjet1_e']
be2fit = data['fitbjet2_e']
be1be2fitpos = fac/be2fit

dicts = data['chi2dict']



indices = []
xAxis = []
values = []

for ds in dicts:
    d = json.loads(ds)
    indices.append(d['index'])
    xAxis.append(d['xAxis'])
    values.append(d['values'])




for i in [1, 4026]:
    fig = plt.figure()


    ax = xAxis[i]
    ay = values[i]

    plt.scatter(ax ,ay, marker='x', label = 'Los')
    plt.axvline(be1meas[i], c='red', label = 'E_B1 measured ' + str(be1meas[i]))
    plt.axvline(be1be2measpos[i], c='orange', label = 'E_B2 measured mapped to Eb1 '+ str(be1be2measpos[i]))
    plt.axvline(be1fit[i], c='magenta', label = 'Eb1 fit')
    plt.legend(fontsize=10)
    plt.xlabel('E_B1')
    plt.ylabel('Los value')


    plt.savefig('chi2forsingleevent' + str(i) + '.png')

    


