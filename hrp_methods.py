# ==============================================================================
# SECTION 1: Core HRP Algorithim with slight revision
# Sourced from 
# 20151227 by MLdP
# Hierarchical Risk Parity
# ==============================================================================
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd

def getIVP(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    cov_ = cov.loc[cItems, cItems]
    w_ = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return cVar

def getQuasiDiag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i, j = df0.index, df0.values - numItems
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(cItems), 2):
            cItems0, cItems1 = cItems[i], cItems[i + 1]
            cVar0, cVar1 = getClusterVar(cov, cItems0), getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha
    return w

def correlDist(corr):
    return ((1 - corr) / 2.) ** 0.5

def min_var(cov):
    icov = np.linalg.inv(np.matrix(cov))
    one = np.ones((icov.shape[0], 1))
    return (icov * one) / (one.T * icov * one)