import numpy as np
import timeit
import numpy.lib.recfunctions as rf  


import scipy.optimize as sopt

from pathlib import Path

# DATAPATH = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22/SKIM_ggF_BulkGraviton_m300/output0.npz')
# data = np.load(DATAPATH)
# events = data['events']


# bigA = np.random.binomial(100,0.75, 10000)
# bigB = np.random.uniform(0,100,10000)
# bigC = np.array(10000*[0])

# def slow(a,b,c):
#     for i in range(10000):
#         c[i] = a[i] + b[i]
#         return c

# def fast(a,b,c):
#     c = a + b
#     return c

# sb = timeit.default_timer()
# s = slow(bigA,bigB,bigC)
# st = timeit.default_timer() - sb

# fb =timeit.default_timer()
# f = fast(bigA,bigB,bigC)
# ft = timeit.default_timer()- fb

# print(st,ft, np.mean(s),np.mean(f))


# testArray = np.random.uniform(0,100,100000)

# class simpleCalc:
#     def __init__(self):
#         self.fs = []
#     def addF(self,function):
#         self.fs.append(function)
#     def calc(self,x):
#         out = []
#         for f in self.fs:
#             out.append(f(x))
#         return out

# myCalc  = simpleCalc()
# myCalc.addF(lambda x : x*x)
# myCalc.addF(lambda x : x+x)
# a = myCalc.calc(testArray)
# print(a)

# seq = [4,3,4,5]
# arr =  [0]

# for n in seq:
#     arr = [n*arr]

# print(arr)

# a = np.linspace(0,9,10)
# b = np.linspace(0,9,10)

# list = [a,b]

# grid = np.meshgrid(*list)

# g=grid[0]

# print(g)

# gc = np.moveaxis(g,[0,1],[1,0])

# ga = gc+a
# gab = np.add(gc,a)

# gb = np.moveaxis(ga,[0,1],[1,0])
# gbb = np.moveaxis(gab,[0,1],[1,0])

# print(gc)
# print(gb)
# print(gbb)

# c = np.arange(len(a))

# print(c-1)


# mf = lambda x,y : x*y

# print(mf(*grid))

# a = np.random.uniform(0,10,100)

# def f(x):
#     return x*a


# b = np.array([[2,1],[3,4]])

# c = np.where(b==np.min(b))

# print(c)

    # a = [['1','2'],['3','4']]

    # print(np.array(a).transpose().tolist())

# a = [1,3,4]

# flist = []

# def make_f(n):
#     def f(x):
#         return x*n
#     return f

# for n in a:
#     flist.append(make_f(n))



# for f in flist:
#     print(f(1))

# a = np.array([1,2,3,4])

# b = np.array([[1,2,3,4,5]]*len(a)).transpose()

# c = a*b


# print(c)

# a= np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
# print(a)

# b = np.transpose(a)

# print(b)

# print(events)

# def f(x):
#     return x**2-1



# a = sopt.fsolve(f,3)

# print(a)

# a = np.array([[1,2,3,4,5,6,5,4,3,2,1],[1,2,3,4,5,6,5,4,3,2,1]])

# b =np.where(a==np.min(a))

# print(b, len(b[0]))

#print(np.sign([-3,0,3]))

# g = np.array([[1,2],[3,4]])



# def edgefinder(grid):
#     s = grid.shape
#     out = np.zeros(s)
#     for i, n in enumerate(s):
#         print(out)
#         print('hi')
#         out = out - np.roll(grid,1,axis=i) + np.roll(grid,-1,axis=i)
#     print(out)

# arr = [1,2,3,4,5,6,7,8,9]
# arr2 = [[1,1,1,0,0,0,0,1,1,1,1],
#         [1,1,1,1,0,0,0,0,1,1,1],
#         [1,1,1,1,1,0,0,0,0,1,1],
#         [1,1,1,1,1,1,0,0,0,0,1],
#         [1,1,1,1,1,1,1,0,0,0,0],
#         [0,1,1,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,1,1,0],
#         [0,0,0,0,1,1,1,1,1,1,1],
#         [1,0,0,0,0,1,1,1,1,1,1],
#         [1,1,0,0,0,0,1,1,1,1,1]]



# x = np.linspace(-45,45,101)

# def f(x):
#     return np.sign(x**2-10)

# y= f(x)

# edgefinder(np.array(arr2))
# edgefinder(np.flip(np.array(arr2)))


# u = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(np.roll(u,(1,1),axis=(0,1)))
# print(u)


# print([float(f) for f in['134','-2134.24','124.44']])

# a = '1234    1234   12434    4562'
# b = a.split(' ')
# c = [f for f in b if f is not '']
# print(c)

# ar = [1,2,3,4,5]
# br = ['1','2','3','4','5']

# for a,b,c in zip(ar,br,ar):
#     print(str(a)+ b + str(c)) 

# arr = [1,2]
# arr += [3,5,6]
# print(arr)



# b = np.array([(1,'ab',np.array([1,2])),(1,'bc',np.array([3,4])),(3,'cd',np.array([5,6]))],dtype=[('number',np.int),('letter','U10'),('array',np.int,(2,))])

# print(b['array'])

# c = np.array([(np.array([7,8])),(np.array([9,10])),(np.array([11,12]))])


# f = np.array([1]*100)

# g =np.empty((100,),np.float64,)

# print(f.shape)
# print(g[0].dtype)
# rf.append_fields(b,'arr',c)
# print(rf.merge_arrays((b,np.transpose(c)),flatten=True))



# a = np.array([1,2,3,4,5,6,7,8])
# print(np.where(a>5))


# import numpy as np
# from pathlib import Path


# path = Path('/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_forTim/SKIM_ggF_BulkGraviton_m300/output_0.npz')
# data = np.load(str(path))
# events = data['events']

# print(np.mean(events['genBQuark1_pt']*np.sinh(events['genBQuark1_eta'])))
# print(np.mean(events['genBQuark2_pt']*np.sinh(events['genBQuark2_eta'])))
# print(np.mean(events['genBQuark1_e']))
# print(np.mean(events['genBQuark2_e']))


# # print(np.std(events['genBQuark1_motherPdgId']))


# import ast
# import json

# array = np.array([0.03032063, 0.0295034,  0.02870817, 0.02793482, 0.02718324, 0.0264533,
#  0.0257449,  0.02505791, 0.02439222, 0.02374772, 0.0231243,  0.02252185,
#  0.02194026, 0.02137942, 0.02083922, 0.02031955, 0.01982031, 0.0193414,
#  0.0188827  ,0.01844413 ,0.01802557 ,0.01762692 ,0.01724808 ,0.01688896,
#  0.01654946 ,0.01622947 ,0.01592891 ,0.01564767 ,0.01538566 ,0.01514278,
#  0.01491895 ,0.01471407 ,0.01452805 ,0.01436079 ,0.01421221 ,0.01408222,
#  0.01397073 ,0.01387765 ,0.01380289 ,0.01374637 ,0.01370801 ,0.01368771,
#  0.01368539 ,0.01370098 ,0.01373438 ,0.01378552 ,0.01385431 ,0.01394068,
#  0.01404454 ,0.01416582 ,0.01430443 ,0.0144603  ,0.01463336 ,0.01482352,
#  0.01503071 ,0.01525485 ,0.01549588 ,0.01575371 ,0.01602828 ,0.01631951,
#  0.01662733 ,0.01695166 ,0.01729244 ,0.0176496  ,0.01802307 ,0.01841278,
#  0.01881866 ,0.01924064 ,0.01967865 ,0.02013264 ,0.02060253 ,0.02108826,
#  0.02158976 ,0.02210697 ,0.02263982 ,0.02318826 ,0.02375221 ,0.02433162,
#  0.02492643 ,0.02553657 ,0.02616199 ,0.02680262 ,0.0274584  ,0.02812928,
#  0.0288152 , 0.02951609, 0.0302319 , 0.03096258, 0.03170807, 0.0324683,
#  0.03324323, 0.0340328 , 0.03483696, 0.03565565, 0.03648881, 0.0373364,
#  0.03819836, 0.03907464, 0.03996518, 0.04086994])

# print(array)

# jsonarrstr = json.dumps(array.tolist())





# print(jsonarrstr)
# print(json.loads(jsonarrstr))

# a = np.linspace(-5,5,10)

# print(np.logical_and(a>2, a <3) )

# RAWDATAPATH = '/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_forTim/SKIM_GGHH_SM/' 
# data = np.load(RAWDATAPATH + 'tet.npz')

# events = data['events']
# # events = np.delete(events, np.where(np.logical_and(events['genLepton1_pdgId'] != 15,events['genLepton1_pdgId'] != 17)))
# # events = np.delete(events, np.where(np.logical_and(events['genLepton4_pdgId'] != 15,events['genLepton4_pdgId'] != 17)))


# print(np.mean(events['genLepton1_pdgId']))
# print(np.mean(events['genLepton2_pdgId']))
# print(np.mean(events['genLepton1_pdgId']+events['genLepton2_pdgId']))
# print(np.std(events['genLepton1_pdgId']+events['genLepton2_pdgId']))
# print(np.mean(events['tauH_mass']))

# a = [[1,2],[3,5]]

# print(a[0][1])

# v = np.array([1,2])

# m = np.array([[3,4],[5,6]])

# print(m.T)

# l = np.array([1,2,3,4])
# l1 = np.array([1,2,3,4])*10
# l2 = np.array([1,2,3,4])*100
# l3 = np.array([1,2,3,4])*1000

# v = np.c_[l,l]

# print(v*l3[:,np.newaxis])


# m = np.dstack((np.c_[l2,l3], np.c_[l,l1]))
# # print(m)
# # print(v)
# # print(np.matmul(m[0],v[0]))
# # print(np.matmul(m,v[0]))

# # print(np.full((10,2,2),m[0]))

# m0 = np.array([[1,0],[0,2]])

# print(m.shape, v.shape)
# a = np.einsum('ji,kj->ki',m0,v)
# print(a)
# print(v)

# print(np.einsum('ij,ij->i',v,a))



# a = np.linspace(5,7,4)

# b=3*a
# c=2*a

# def combineArrayToMatrixArray(a,b,c,d):
#     out =np.dstack((np.c_[a,b], np.c_[c,d]))

#     return out

# def inverse2x2(A):

#     det =  A[0][0]*A[1][1]-A[0][1]*A[1][0]
#     return np.array([
#         [A[1][1],-A[1][0]],
#         [-A[0][1],A[0][0]]]
#         )/det


# covMat = combineArrayToMatrixArray(a,b ,b ,c)
# inverseCovMat = np.array([inverse2x2(m) for m in covMat])

# print(inverse2x2(np.array([[a[0],b[0]],[b[0],c[0]]])))

# print(covMat)
# print('\n \n \n')
# print(inverseCovMat)

# print(np.matmul(covMat,inverseCovMat))

# print(np.mean(inverseCovMat[:][0][0]))
# print(np.mean(inverseCovMat[:][0][1]))
# print(np.mean(inverseCovMat[:][1][0]))
# print(np.mean(inverseCovMat[:][1][1]))

# data = np.load('/nfs/dust/cms/user/lukastim/bachelor/data/DataSet.npy')






# print(np.mean(data['met_invcov00']))
# print(np.mean(data['met_invcov01']))
# print(np.mean(data['met_invcov11']))

# print(min(data['met_invcov00']))
# print(min(data['met_invcov01']))
# print(min(data['met_invcov11']))

# print(np.std(data['met_invcov00']))
# print(np.std(data['met_invcov01']))
# print(np.std(data['met_invcov11']))



# print(np.mean(data['met_cov00']))
# print(np.mean(data['met_cov01']))
# print(np.mean(data['met_cov11']))
# print(np.std(data['met_cov00']))
# print(np.std(data['met_cov01']))
# print(np.std(data['met_cov11']))

# def f(x):
#     return (x[0]*x[1]+x[2],0,0)

# start = np.array([3,2,1])

# print(sopt.fsolve(f,start))
# print(sopt.fsolve(f,start))
# print(sopt.root(f,start))

# a = np.linspace(0,99,100)
# b = np.reshape(a,(10,10))

# print(np.amin(b,1))
# print(np.amin(b,0))

# print(a[2:5])

# 4,63065383e+00
# 4,46372918e+00
# 4,29935300e+00
# 4,13752529e+00
# 3,97824604e+00
# 3,82151526e+00
# 3,66733295e+00
# 3,51569911e+00
# 3,36661374e+00
# 3,22007684e+00
# 3,07608840e+00
# 2,93464843e+00
# 2,79575693e+00
# 2,65941389e+00
# 2,52561933e+00
# 2,39437323e+00
# 2,26567560e+00
# 2,13952644e+00
# 2,01592574e+00
# 1,89487352e+00
# 1,77636976e+00
# 1,66041447e+00
# 1,54700765e+00
# 1,43614929e+00
# 1,32783941e+00
# 1,22207799e+00
# 1,11886504e+00
# 1,01820055e+00
# 9,20084540e-01
# 8,24516993e-01
# 7,31497914e-01
# 6,41027303e-01
# 5,53105160e-01
# 4,67731485e-01
# 3,84906278e-01
# 3,04629538e-01
# 2,26901267e-01
# 1,50277006e-01
# 7,50972020e-02
# 2,46586634e-03
# -6,76170014e-02
# -1,35151401e-01
# -2,00137333e-01
# -2,62574797e-01
# -3,22463793e-01
# -3,79804321e-01
# -4,34596381e-01
# -4,86839974e-01
# -5,36535098e-01
# -5,83681755e-01
# -6,28279943e-01
# -6,70329664e-01
# -7,09830916e-01
# -7,46783701e-01
# -7,81188018e-01
# -8,13043867e-01
# -8,42351248e-01
# -8,69110161e-01
# -8,93320606e-01
# -9,14982584e-01
# -9,34096093e-01
# -9,50661135e-01
# -9,64677708e-01
# -9,76145814e-01
# -9,85065451e-01
# -9,91436621e-01
# -9,95259323e-01
# -9,96533557e-01
# -9,97614378e-01
# -9,98327130e-01
# -9,98941976e-01
# -9,99463818e-01
# -9,99850304e-01
# -1,00000000e+00
# -9,99774099e-01
# -9,99557832e-01
# -9,98394762e-01
# -9,97120528e-01
# -9,94787232e-01
# -9,90964531e-01
# -9,86828451e-01
# -9,80457281e-01
# -9,72713718e-01
# -9,63794081e-01
# -9,52325975e-01
# -9,38309401e-01
# -9,24037411e-01
# -9,07472369e-01
# -8,88358860e-01
# -8,66696882e-01
# -8,42486437e-01
# -8,15727524e-01
# -7,86893349e-01
# -7,57585968e-01
# -7,25730119e-01
# -6,91325803e-01
# -6,54373018e-01
# -6,14871765e-01
# -5,72822044e-01
# -5,28223856e-01









# 5.35562101  5.20893778  5.06401124  4.92084138  4.7794282   4.63977171
# 4.50187191  4.36572879  4.2305215   4.09613507  3.96350532  3.83263226
# 3.70351588  3.57615619  3.45055318  3.32670686  3.20461723  3.08428428
# 2.96570801  2.84888843  2.73382554  2.62051933  2.50896981  2.39917697
# 2.29114082  2.18486135  2.08033857  1.97757248  1.87656306  1.77731034
# 1.6798143   1.58407495  1.49009228  1.39786629  1.30739699  1.21868438
# 1.13172845  1.04652921  0.96308666  0.88140078  0.8014716   0.7232991
# 0.64688328  0.57222415  0.49932171  0.42817595  0.35878688  0.29115449
# 0.22527879  0.16115977  0.09879744  0.03819179 -0.02065717 -0.07774944
# -0.13308503 -0.18666394 -0.23848616 -0.28855169 -0.33686054 -0.3834127
# -0.42820818 -0.47124697 -0.51252908 -0.5520545  -0.58982324 -0.62583529
# -0.66009066 -0.69258934 -0.72333133 -0.75231664 -0.77954526 -0.8050172
# -0.82873245 -0.85069102 -0.87093165 -0.89113354 -0.90957873 -0.92626724
# -0.94119907 -0.95437421 -0.96579266 -0.97545443 -0.98335952 -0.98950792
# -0.99389963 -0.99653466 -0.997413   -0.99762389 -0.99810361 -0.99861499
# -0.99935392 -1.         -0.99912166 -0.99648663 -0.99209492 -0.98594652
# -0.97804143 -0.96837966 -0.95696121 -0.94378607

# a = np.arange(0,1000,1)

# a2 = np.reshape(a,(5,20,10))

# b = np.arange(0,200,1)
# b2 = np.reshape(b,(20,10))

# a3 = np.moveaxis(a2,0,1)


# c=  b2 + a2 
# # d= a3 + b


# print(b2,b2.shape)
# # print(d)

# a = np.array([1,2,3,4,2])

# print(np.where(min(a)==a)[0])


invCovMat = np.array([[2,-1],[3,-2]])

delta = np.reshape(np.linspace(1,50,50),(5,5,2))


print(delta)

print(np.einsum('ilj,ilj->il',delta, np.einsum('ji,klj->kli',invCovMat,delta)))