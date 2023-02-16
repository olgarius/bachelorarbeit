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



a = np.array([1,2,3,4,5,6,7,8])
print(np.where(a>5))
