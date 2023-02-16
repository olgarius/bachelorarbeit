import numpy as np
import matplotlib.pyplot as plt
import plotting as p

from pathlib import Path
import util as u

DATAPATH = Path('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet.npy')
DATAPATH2  = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')

MHIGGS = 125.25

data = np.load(DATAPATH2)
events2 = data['events']
b1eGen = u.structToArray(events2,'genBQuark1_e' )
b2eGen = u.structToArray(events2,'genBQuark2_e' )
facGen  = u.getFacs(events2,'genBQuark1_phi','genBQuark1_eta', 'genBQuark2_phi','genBQuark2_eta')
sfacGen = np.sqrt(facGen)




xGen =  b1eGen / sfacGen
yGen =  b2eGen /sfacGen

fig = p.plot2d2(xGen,yGen,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)

#load data and select relevant data
events = np.load(DATAPATH)

chi2val = u.structToArray(events,'chi2val')
indexchi2Smaller0d5 = np.where(chi2val<0.5)[0]
indexchi2Smaller2d5 = np.where(chi2val<2.5)[0]
indexchi2Smaller10 = np.where(chi2val<10)[0]





b1e = u.structToArray(events,'bjet1_e' )
b2e = u.structToArray(events,'bjet2_e' )
fac  = u.getFacs(events,'bjet1_phi','bjet1_eta', 'bjet2_phi','bjet2_eta')
sfac = np.sqrt(fac)

t = np.linspace(1,10,30)

fx = 1/t
fy = 1/fx

plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestGen.png')


print(len(sfac))

x =  b1e / sfac
y =  b2e /sfac

fig = p.plot2d2(x,y,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTest.png')








xComp0d5 = np.take(x,indexchi2Smaller0d5)
xComp2d5 = np.take(x,indexchi2Smaller2d5)
xComp10 = np.take(x,indexchi2Smaller10)

yComp0d5 = np.take(y,indexchi2Smaller0d5)
yComp2d5 = np.take(y,indexchi2Smaller2d5)
yComp10 = np.take(y,indexchi2Smaller10)

fig = p.plot2d2(xComp0d5,yComp0d5,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestComp0d5.png')

fig = p.plot2d2(xComp2d5,yComp2d5,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestComp2d5.png')

fig = p.plot2d2(xComp10,yComp10,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestComp10.png')

xGenComp0d5 = np.take(xGen,indexchi2Smaller0d5)
xGenComp2d5 = np.take(xGen,indexchi2Smaller2d5)
xGenComp10 = np.take(xGen,indexchi2Smaller10)

yGenComp0d5 = np.take(yGen,indexchi2Smaller0d5)
yGenComp2d5 = np.take(yGen,indexchi2Smaller2d5)
yGenComp10 = np.take(yGen,indexchi2Smaller10)

fig = p.plot2d2(xGenComp0d5,yGenComp0d5,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestGenComp0d5.png')

fig = p.plot2d2(xGenComp2d5,yGenComp2d5,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestGenComp2d5.png')

fig = p.plot2d2(xGenComp10,yGenComp10,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)



plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')

plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestGenComp10.png')









sigmab2 = u.structToArray(events,'bjet2_sigma')*b2e

low = b2e - sigmab2
high = b2e + sigmab2

# low2 = 0.7*b1e
# high2 = 1.3*b1e

b1etob2e = fac/b1e
# b1etob2elow = fac/high2
# b1etob2ehigh = fac/low2

cleanx = []
cleany = []
# cleanx2 = []
# cleany2 = []

for i, b in enumerate(b1etob2e):
    if  low[i] < b and high[i] > b:
        cleanx.append(x[i])
        cleany.append(y[i])

# for i, b in enumerate(b2e):
#     if  b1etob2elow[i] < b and b1etob2ehigh[i] > b:
#         cleanx2.append(x[i])
#         cleany2.append(y[i])

fig = p.plot2d2(cleanx,cleany,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=100)
plt.scatter(fy,fx, c = 'r', marker='x', s = 4)
plt.scatter(fx,fy, c = 'r', marker='x', s = 4)

plt.xlabel('bjet1_e / f')
plt.ylabel('bjet2_e / f')
plt.legend(['1/x'], fontsize= 'xx-small')
plt.suptitle('f = sqrt(m_H²/(2*(1-cos(alpha))))')
plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestCleaned.png')
print(len(cleanx))

# fig = p.plot2d2(cleanx2,cleany2,'/afs/desy.de/user/l/lukastim/code/bachelorarbeit/','f_bjet1_e', 'f_bjet2_e',xlim=[0,10],ylim=[0,10],bins=1000)
# plt.scatter(fy,fx, c = 'r')
# plt.scatter(fx,fy, c = 'r')
# plt.savefig('/afs/desy.de/user/l/lukastim/code/bachelorarbeit/'+'dataTestCleaned2.png')
# print(len(cleanx2))






