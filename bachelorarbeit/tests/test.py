import numpy as np
import bachelorarbeit.grid as grid
import bachelorarbeit.mathutil as u
import plotting

import multiprocessing as mp

def testGrid():
    def f1(x):
        print(x)
        return 1000*x*x

    def f2(x):
        return 0

    def f3(x,y):
        return 0

    g = grid.grid()

    x = np.arange(0,6)
    y = np.arange(-5,0)


    g.addDimension(x)
    g.addDimension(y)
    

    g.addFunction(f1,[1])
    g.addFunction(f2,[2])
    g.addFunction(f3,[1,2])

    g.evaluate()

    e = g.evaluated
    m = g.getMin()
    p = g.getMinCoords()

    print(e)
    print(m)
    print(p)
    plotting.plot2d3(e,g.evaValues,'bachelorarbeit/','x','y',[-6,6],[-6,6])

def testUtil():
    p1 = (1,np.pi,1)
    p2 = (1,np.pi,0)

    alpha = u.angleBetweenVectorsInPolarCoordinats(p1,p2)
    
    print(alpha)

# testUtil()

testGrid()