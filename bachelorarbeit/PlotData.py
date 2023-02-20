import plotting 
import util

import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path





#DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_GluGluHToTauTau/output0.npz')
DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')
DATAPATH2 = Path('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet.npy')
PATHTOBA = '/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'
PATHTOPLOTS = PATHTOBA + 'plots/'
PATHTOPLOTS2D =  PATHTOPLOTS + 'plots2d/'
PATHTOPLOTS1D =  PATHTOPLOTS + 'plots1d/'
PATHTOPLOTSDATVAL = PATHTOPLOTS + 'datavalidation/' 

BINSIZE = 10
XLIM = 2000
YLIM = 10000

tag = 'bH_mass'
tags = ['dau2_pt', 'dau2_eta', 'dau2_phi', 'dau2_e','dau1_pt', 'dau1_eta', 'dau1_phi', 'dau1_e','tauH_mass', 'bjet1_pt', 'bjet1_eta', 'bjet1_phi', 'bjet1_e', 'bjet2_pt', 'bjet2_eta', 'bjet2_phi', 'bjet2_e', 'bH_mass']


#load Data and select events
data = np.load(DATAPATH)
data2= np.load(DATAPATH2)

events = data['events']


# def selectData(keylist, structarray):
    
#     dataList = []
#     dtypeList = []

#     for k in keylist:
#         value = structarray[k]
#         dataList.append(value)
#         dtypeList.append((k,value.dtype.str))

#     np.concat()   

def selectData(keylist, structarray):
    return structarray[keylist]

# val = selectData([tag],events)

# a = np.array([k[0] for k in val])


# plt.hist(a,bins=np.arange(min(a),max(a)+BINSIZE,BINSIZE))
# plt.xlim([0,XLIM])
# plt.ylim([0,YLIM])

# # plt.yscale('log')


# plt.savefig(tag + '.png')


# for t in tags:
#     values = selectData([t],events)
#     a = np.array([k[0] for k in values])
    
#     fig = plt.figure()

#     if '_eta' in t: 
#         binsize = 0.1    
#         # xlim = 2*np.pi
#         # ylim = 10000
#     elif '_phi' in t:
#         binsize = 0.1    
#         # xlim = np.pi
#         # ylim = 10000
#     elif '_e' or '_pt' in t:
#         binsize = 10
#         # xlim = 1000
#         # ylim = 20000
#     elif '_mass' in t:
#         binsize = 10
#         # xlim = 1000
#         # ylim = 20000


#     print(t)
#     h, b = np.histogram(a,bins=np.arange(min(a),max(a)+binsize,binsize))

#     hep.histplot(h,b)
#     # plt.xlim([0,xlim])
#     # plt.ylim([0,ylim])

#     plt.savefig('bachelorarbeit/plots/' + t + '.png')






# plotting.plot2d('dau1_e', 'dau1_pt', events, PATHTOPLOTS2D, xlim=[0,300], ylim=[0,300], bins=500)
# plotting.plot2d('bjet1_e', 'bjet1_pt', events, PATHTOPLOTS2D, xlim=[0,300], ylim=[0,300], bins=1000)
# plotting.plot2d('bH_mass', 'tauH_mass', events, PATHTOPLOTS2D, xlim=[0,300], ylim=[0,300], bins=1000)
# plotting.plot2d('bjet1_e', 'dau1_e', events, PATHTOPLOTS2D, xlim=[0,300], ylim=[0,300], bins=1000)
# plotting.plot2d('bjet1_eta', 'bjet1_e', events, PATHTOPLOTS2D, xlim=[- np.pi, np.pi], ylim=[0,300], bins=500)
# plotting.plot2d('dau1_eta', 'dau1_e', events, PATHTOPLOTS2D, xlim=[- np.pi, np.pi], ylim=[0,300], bins=500)
# plotting.plot2d('bjet1_eta', 'bjet1_pt', events, PATHTOPLOTS2D, xlim=[- np.pi, np.pi], ylim=[0,300], bins=500)
# plotting.plot2d('dau1_eta', 'dau1_pt', events, PATHTOPLOTS2D, xlim=[- np.pi, np.pi], ylim=[0,300], bins=500)
# plotting.plot2d('bH_e', 'tauH_e', events, PATHTOPLOTS2D, xlim=[0,300],ylim=[0,300], bins=1000)

# plotting.plot1d1('bjet2_e', events, './', xlim=[0,1000])
# plotting.plot1d1('dau2_e', events, './', xlim=[0,1000])


#plotting.plot1d3(np.array([k[0] for k in events[['bjet2_e']]]),np.load(PATHTOPLOTS + '/bjet2_e_fitted.npy'),'./','bE')
#plotting.plot1d3(np.array([k[0] for k in events[['dau2_e']]]),np.load(PATHTOPLOTS + '/dau2_e_fitted.npy'),'./','dauE')

# plotting.plot2d('bjet1_e', 'bjet2_e', events, PATHTOPLOTS2D, xlim=[0,300], ylim=[0,300], bins=1000)
# plotting.plot1d1('bjet1_e', events, PATHTOPLOTS , xlim=[0,1000])

mins = np.load(PATHTOBA+ 'chi2mins.npy')
meas = util.structToArray(data2,'genBQuark1_e')

# plotting.plot1d2(mins,PATHTOPLOTS,'bjet1_e_fitted',xlim=[0,1000])

# fig = plt.figure()
# plt.scatter(meas,mins)
# plt.savefig(PATHTOPLOTS + 'realVSfit.png')


# plotting.plot2d2(meas,mins, PATHTOPLOTSDATVAL,'real_bjet1_e','fitvalue')


# grids = np.load(PATHTOPLOTS + 'chi2.npy')
# fig = plt.figure()
# plt.plot(grids[0])
# plt.savefig(PATHTOPLOTS + 'chi2plot1.png')

be1fit = util.structToArray(data2,'fitbjet1_e')
be2fit = util.structToArray(data2,'fitbjet2_e')

be1 = util.structToArray(data2,'bjet1_e')
pt1 = util.structToArray(data2,'bjet1_pt')
eta1 = util.structToArray(data2,'bjet1_eta')
be2 = util.structToArray(data2,'bjet2_e')
be1gen = util.structToArray(data2,'genBQuark1_e')
be2gen = util.structToArray(data2,'genBQuark2_e')
fac = util.structToArray(data2,'bie_to_bje')



chi2sigma = util.structToArray(data2,'fitbjet1_esigma')

chi2val = util.structToArray(data2,'chi2val')
indexchi2Smaller0d5 = np.where(chi2val<0.5)[0]
indexchi2Smaller2d5 = np.where(chi2val<2.5)[0]
indexchi2Smaller10 = np.where(chi2val<10)[0]





pullFit = (be1fit-meas)/meas
pullOriginal = (be1 - meas)/meas

pullFitComp0d5 = (np.take(pullFit,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
pullFitComp2d5 = (np.take(pullFit,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
pullFitComp10 = (np.take(pullFit,indexchi2Smaller10),r"$\chi^2 < 10$ ")


pullOriginalComp0d5 = (np.take(pullOriginal,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
pullOriginalComp2d5 = (np.take(pullOriginal,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
pullOriginalComp10 = (np.take(pullOriginal,indexchi2Smaller10),r"$\chi^2 < 10$ ")

chi2sigmaComp0d5 = (np.take(chi2sigma,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
chi2sigmaComp2d5 = (np.take(chi2sigma,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
chi2sigmaComp10 = (np.take(chi2sigma,indexchi2Smaller10),r"$\chi^2 < 10$ ")



plotting.plotHist(chi2sigma,PATHTOPLOTSDATVAL,r"$\sigma_{\chi^2}$",r"$\sigma_{\chi^2}$-Distribution", [0,500], 100, chi2sigmaComp0d5, chi2sigmaComp2d5, chi2sigmaComp10,ylim=[0,400] ,alttitle='sigmaChi2Distribution', density=False , yLabel=r"Amount")


plotting.plotHist(pullFit, PATHTOPLOTSDATVAL,r"$\frac{E_{B1}^{\textit{\small{fit}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$", r"Pull Test",[-2.5,12.5],200,pullFitComp0d5, pullFitComp2d5, pullFitComp10, ylim=[0.8,800], density=False , yLabel=r"Amount", yscale='log')
plotting.plotHist(pullOriginal, PATHTOPLOTSDATVAL,r"$\frac{E_{B1}^{\textit{\small{meas}}}-E_{B1}^{\textit{\small{gen}}}}{E_{B1}^{\textit{\small{gen}}}}$",r"Pull Test Original Values",[-2.5,12.5],200,pullOriginalComp0d5, pullOriginalComp2d5, pullOriginalComp10, ylim=[0.8,800], density=False , yLabel=r"Amount", yscale='log')

plotting.plotHist(chi2val,PATHTOPLOTSDATVAL,r"$\chi^2$-Value",r"$\chi^2$-Value Distribution",xlim=[0,4], ylim =[0.01,10], yscale='log',bins=80, alttitle='Chi2ValueDistribution')

# plotting.plot1d2(v2,PATHTOPLOTSDATVAL,'(be2-gen)div(gen)',xlim=[-10,10],binsize=0.1,scale='linear')
# plotting.plot1d2(v,PATHTOPLOTSDATVAL,'(befit-gen)div(gen)',xlim=[-10,10],binsize=0.1,scale='linear')
# plotting.plot1d2(v,PATHTOPLOTSDATVAL,'(befit-gen)div(gen)log',xlim=[-50,50],binsize=2)
# plotting.plot1d2(v2,PATHTOPLOTSDATVAL,'(be2-gen)div(gen)log',xlim=[-50,50],binsize=2)


pt1Comp0d5 = np.take(pt1,indexchi2Smaller0d5)
pt1Comp2d5 = np.take(pt1,indexchi2Smaller2d5)
pt1Comp10 = np.take(pt1,indexchi2Smaller10)

eta1Comp0d5 = np.take(eta1,indexchi2Smaller0d5)
eta1Comp2d5 = np.take(eta1,indexchi2Smaller2d5)
eta1Comp10 = np.take(eta1,indexchi2Smaller10)

fig = plotting.plot2d2(pt1Comp0d5,eta1Comp0d5,PATHTOPLOTSDATVAL,'pt','eta',[0,300],[-5,5], bins=100)
plt.savefig('pt_eta_comp0d5.png')
fig = plotting.plot2d2(pt1Comp2d5,eta1Comp2d5,PATHTOPLOTSDATVAL,'pt','eta',[0,300],[-5,5], bins=100)
plt.savefig('pt_eta_comp2d5.png')
fig =  plotting.plot2d2(pt1Comp10,eta1Comp10,PATHTOPLOTSDATVAL,'pt','eta',[0,300],[-5,5], bins=100)
plt.savefig('pt_eta_comp10.png')


mde = np.sqrt(be1**2-pt1**2)/be1
mdeComp0d5 = (np.take(mde,indexchi2Smaller0d5),r"$\chi^2 < 0.5$ ")
mdeComp2d5 = (np.take(mde,indexchi2Smaller2d5),r"$\chi^2 < 2.5$ ")
mdeComp10 = (np.take(mde,indexchi2Smaller10),r"$\chi^2 < 10$ ")
plotting.plotHist(mde, PATHTOPLOTSDATVAL,r"$\frac{\sqrt{p_{TB1}^2-E_{B1}^2}}{E_{B1}}$", r"Masse Verteilung",[0,1],200,mdeComp0d5, mdeComp2d5, mdeComp10, ylim=[1,1000], density=False , yLabel=r"Amount", yscale='log')

sfac = np.sqrt(fac)

be1fitdfac = be1fit/sfac
be2fitdfac = be2fit/sfac

be1dfac = be1/sfac
be2dfac = be2/sfac
be1gendfac = be1gen/sfac
be2gendfac = be2gen/sfac



be1dfacComp0d5 = np.take(be1dfac,indexchi2Smaller0d5)
be1dfacComp2d5 = np.take(be1dfac,indexchi2Smaller2d5)
be1dfacComp10 = np.take(be1dfac,indexchi2Smaller10)

be2dfacComp0d5 = np.take(be2dfac,indexchi2Smaller0d5)
be2dfacComp2d5 = np.take(be2dfac,indexchi2Smaller2d5)
be2dfacComp10 = np.take(be2dfac,indexchi2Smaller10)

be1gendfacComp0d5 = np.take(be1gendfac,indexchi2Smaller0d5)
be1gendfacComp2d5 = np.take(be1gendfac,indexchi2Smaller2d5)
be1gendfacComp10 = np.take(be1gendfac,indexchi2Smaller10)

be2gendfacComp0d5 = np.take(be2gendfac,indexchi2Smaller0d5)
be2gendfacComp2d5 = np.take(be2gendfac,indexchi2Smaller2d5)
be2gendfacComp10 = np.take(be2gendfac,indexchi2Smaller10)


t = np.linspace(1,10,30)

fx = 1/t
fy = 1/fx

fxx = np.append(fy,fx)
fyy = np.append(fx,fy)
f = [fxx,fyy]

plotting.plot1v2hist(be1fitdfac,be2fitdfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{fit}}{f}$",r"$\frac{E_{B2}^{fit}}{f}$",r" $E_{B2}^{fit}$ vs $ E_{B2}^{fit}$ ", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha)})}}$",'SanityCheck')


plotting.plot1v2hist(be1dfac,be2dfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ ", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(alpha)})}}$",'BE1vsBE2')
plotting.plot1v2hist(be1dfacComp0d5,be2dfacComp0d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100,r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 0.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller0d5')
plotting.plot1v2hist(be1dfacComp2d5,be2dfacComp2d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 2.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller2d5')
plotting.plot1v2hist(be1dfacComp10,be2dfacComp10, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100,  r"$\frac{E_{B2}^{meas}}{f}$",r"$\frac{E_{B2}^{meas}}{f}$",r" $E_{B2}^{meas}$ vs $ E_{B2}^{meas}$ for $\chi^2 < 10$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'BE1vsBE2Chi2smaller10')

plotting.plot1v2hist(be1gendfac,be2gendfac, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ ", PATHTOPLOTSDATVAL, r" $f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(alpha)})}}$",'genBE1vsBE2')
plotting.plot1v2hist(be1gendfacComp0d5,be2gendfacComp0d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 0.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller0d5')
plotting.plot1v2hist(be1gendfacComp2d5,be2gendfacComp2d5, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 2.5$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller2d5')
plotting.plot1v2hist(be1gendfacComp10,be2gendfacComp10, f,r"$y=\frac{1}{x}$",[0,10],[0,10], 100, r"$\frac{E_{B2}^{gen}}{f}$",r"$\frac{E_{B2}^{gen}}{f}$",r" $E_{B2}^{gen}$ vs $ E_{B2}^{gen}$ for $\chi^2 < 10$", PATHTOPLOTSDATVAL, r"$f = \sqrt{\frac{m_H²}{2 \cdot (1-\cos{(\alpha}))}}$",'genBE1vsBE2Chi2smaller10')



