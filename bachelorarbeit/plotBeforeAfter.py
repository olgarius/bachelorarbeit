import plotting 
import structutil as su

import matplotlib.pyplot as plt

import numpy as np


from PATHS import WORKINGDATAPATH
from PATHS import PATHTOPLOTS1D
from PATHS import PATHTOPLOTS2D
from PATHS import PATHTOPLOTSDATVAL
from PATHS import PATHTOPLOTSBJET
from PATHS import PATHTOPLOTSTAU



data2= np.load(WORKINGDATAPATH)

b1meas = su.structToArray(data2,'bjet1_e')
b1fit = su.structToArray(data2,'fitbjet1_e')
b1gen = su.structToArray(data2,'genBQuark1_e')

b1measpull = b1meas/b1gen - 1
b1fitpull = b1fit/b1gen - 1


dau1meas = su.structToArray(data2,'dau1_e')
dau1fit = su.structToArray(data2,'fitdau1_e')
dau1gen = su.structToArray(data2,'genLepton1_e')

dau1measpull = dau1meas/dau1gen - 1
dau1fitpull = dau1fit/dau1gen - 1

plotting.plotHistCompare(PATHTOPLOTSBJET, r"Pull Value", r"Fitted vs. Measuerd $E_{B1}$ Values", [-1,1], 80, (b1measpull, r"$\frac{E_{B1}^{m}}{E_{B1}^{g}}-1$ "),(b1fitpull,r"$\frac{E_{B1}^{f}}{E_{B1}^{g}}-1$ "),ylim=[0,3], alttitle='BPullFitVsMeas')
plotting.plotHistCompare(PATHTOPLOTSTAU, r"Pull Value", r"Fitted vs. Measuerd $E_{\tau 1}$ Values", [-1,1.5], 100, (dau1measpull, r"$\frac{E_{\tau 1}^{m}}{E_{\tau 1}^{g}}-1$ "),(dau1fitpull,r"$\frac{E_{\tau 1}^{f}}{E_{\tau 1}^{g}}-1$ "),ylim=[0,2], alttitle='TauPullFitVsMeas')

plotting.scatter(PATHTOPLOTSBJET, r"B-Jet Energie Comparison to Generator-Values", r"$E_{B1}^g$ in GeV",r"$E_{B1}$ in GeV",(b1gen,b1meas,r"$E_{B1}^m$"),(b1gen,b1fit,r"$E_{B1}^f$"))
plotting.scatter(PATHTOPLOTSTAU, r"Tau Comparison to Generator-Values", r"$E_{\tau 1}^g$ in GeV",r"$E_{\tau 1}$ in GeV",(dau1gen,dau1meas,r"$E_{\tau 1}^m$"),(dau1gen,dau1fit,r"$E_{\tau 1}^f$"))

