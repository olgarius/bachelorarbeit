import numpy as np


import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import mplhep as hep





class losTerm:
    def __init__(self, yMeas, sigma):
        self.yMeas = yMeas
        self.sigma = sigma

    def getTerm(self, yfit):
        return ((yfit - self.yMeas)/self.sigma)^2


class grid:
    def __init__(self):
        self.dimension = 0          #dimension counter
        self.functions = []         #List of function on which the grid should be evaluated on (additive).
        self.fdimensions = []       #List of Dimensions of each function
        self.evaValues = []         #List of Values for each axis on which the function depending on that dimenson should be evaluated on.
        self.evaluated = None       #The Values of the grid after evaluation.
    
    
    def addDimension(self, evaValues): 
        """
        Adds a Dimension to the Grid.

        Parameters
        ----------

        evaValues : array_like
            The coordinates on this axis where functions should be evaluated.     
        
        """
        self.dimension += 1
        self.evaValues.append(evaValues)



    def addFunction(self, function, dimensions):
        """
        Adds a function to the grid. The function is additiv to all other functions given.

        Parameters
        ----------

        function : function
            Function of one more dimensions of the grid.
        dimensions : list
            List of dimensions on which the function should be evaluated on. The number of dimensions has to match the number of arguments of the function!
        
        """
        #check if dimension exists 
        if max(dimensions) <= self.dimension:
            #add function and the dimension on which teh function depends
            self.functions.append(function)
            self.fdimensions.append(dimensions)
        else:
            print("No function was added, some dimensions do not exist")


    

    def buildGrid(self,dimensions):
        """
        Builds and returns an evaluation grid over the given dimensions.
        The grid values are the to the dimension belonging evaValues, which where added with their dimension. 

        Parameters
        ----------

        dimensions : list
            List containig the dimension numbers over which the grid is created.  

        """

        #collect evaluation values of all dimenisons given
        x = []
        for n in dimensions:
            x.append(self.evaValues[n-1])
        #make grid and return it
        return  np.meshgrid(*x)

    def mapDimensions(self, dimlist, workingdims):
        mapped = []
        for d in dimlist:
            mapped.append(workingdims.index(d))
        return mapped


    def evaluate(self, dimList = None):
        """
        Evalauates the grid in the given dimensions. Only function which are dependend on these dimensions are taken into account.

        Parameters
        ----------

        dimlist : list
            List of dimensions on which the grid should be evaluated. If no list is given, the grid is evaluated in all dimensions.

        """
        #check if dimensions are given
        if dimList is None:
            dimList = list(range(1,self.dimension+1))


        #build an evaluation grid with the specified dimensions
        revDimList = list(range(len(dimList)))
        #revDimList.reverse()
        dimSet = set(dimList)


        for n in revDimList:
            if n is revDimList[0]:
                evaOutList = [len(self.evaValues[n-1])*[0]]
            else:
                evaOutList = [len(self.evaValues[n-1])*evaOutList]
        evaOut = np.array(evaOutList[0])
    
        


        #evaluate all functions dependend on the given dimensions and add them up
        for i,f in enumerate(self.functions):
            fdims = self.fdimensions[i] #get dimensions of function
            if set(fdims) <= dimSet:   #evaluate function if dependet on given dimensions
                fevaValues = self.buildGrid(fdims)  #get evaluation values
                values = f(*fevaValues)  #calculate function values
                
                source = np.array(self.mapDimensions(fdims,dimList))-1
                target = np.arange(len(fdims))

                #add evaluated values to the output
                tempEva = np.moveaxis(evaOut,source,target) 
                temp2Eva = tempEva + values
                evaOut = np.moveaxis(temp2Eva,target,source)
                
        self.evaluated = evaOut

    def getMin(self):
        return np.where(self.evaluated==np.min(self.evaluated))

    def getMinCoords(self):
       m = self.getMin()
       out = []
       for i,n in enumerate(m):
            out.append(self.evaValues[i][n[0]])
       return out

    def getCut(self, hight):
        grid = np.sign(self.evaluated - hight)

    




        

                
        
        
            
            
        

