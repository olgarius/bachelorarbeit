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
        newdata = rf.append_fields(olddata,key,data, usemask=False)
    return newdata

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
