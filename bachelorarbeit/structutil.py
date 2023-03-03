import numpy as np
import numpy.lib.recfunctions as rf  


def structToArray(structarray,key):
    return np.array([k[0] for k in structarray[[key]]])

def updateDataSet(path,key,data):
    olddata = np.load(path)
    newdata = updateStructArray(olddata, key, data)
    np.save(path,newdata.data)

def updateStructArray(olddata, key, data):
    keys = olddata.dtype.names 
    if key in keys:
        olddata[key] = data
        newdata = olddata
    else:
    
        newdata = rf.append_fields(olddata,key,data ,usemask=False)
    return newdata

def updateStructArrayWithFloatArray(olddata, key, data):
    keys = olddata.dtype.names 
    if key in keys:
        olddata[key] = data
        newdata = olddata
    else:
        dt = np.dtype([(key, float ,data[0].shape)])
        new = np.array(data, dt)
        newdata = rf.merge_arrays([olddata, new])
    return newdata

def updateDataSetWithFloatArray(path,key,data):
    olddata = np.load(path)
    newdata = updateStructArrayWithFloatArray(olddata, key, data)
    np.save(path,newdata.data)




def getIndexForCondition(data, key, condition, comparevalue):
    values = structToArray(data, key)
    indices = []
    for i, v in enumerate(values):
        if condition(v, comparevalue):
            indices.append(i)
    return indices

def removeEvents(data, indexList):
    indexSet = list(set(indexList))
    return np.delete(data,indexSet)

def switchEntries(structArray,key1,key2, mask):
    temp1 = structToArray(structArray,key1)
    temp2 = structToArray(structArray,key2)
    structArray[key1] =  np.where(mask == 1, temp2 , temp1)
    structArray[key2] =  np.where(mask == 1, temp1 , temp2)

    return structArray

