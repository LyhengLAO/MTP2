#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from os.path import join


# In[ ]:


def load_data():
    ret = pd.read_csv('data/ret.csv', header = None).values 
    #ret.shape = (10344, 3251), (day, stock)
    univ = pd.read_csv('data/topMV95.csv', header = None).values 
    univ -= 1 #because Matlab is 1 indexed
    #univ.shape = (360, 1000), (OOS month, sorted list of stocks to consider for that period)
    dates = pd.read_csv('data/mydatestr.txt', header = None, parse_dates = [0]) 
    #dates.shape = (10344, 1), date for each day in return (not a numpy array, but a dataframe with DT objects)
    tradeidx = pd.read_csv('data/investDateIdx.csv', header = None).values 
    tradeidx -= 1 #because Matlab is 1 indexed!
    #tradeidx.shape = (360, 1), (row of univ -> index in ret matrix)
    ret[ret == -500] = np.nan
    ret = ret / 100 #ret is in percent
    ret_nonan = ret.copy()
    ret_nonan[np.isnan(ret)] = 0

    return ret, ret_nonan, univ, tradeidx, dates


# In[ ]:


def kendall_cov(data):
    df = pd.DataFrame(data)
    kendall_corr_mat = df.corr(method='kendall').values
    corr_mat = np.sin(0.5 * np.pi * kendall_corr_mat)
    stdmat = np.diag(np.sqrt(np.diag(np.cov(data.T))))
    return stdmat.dot(corr_mat).dot(stdmat)


# In[ ]:


def get_invest_period(h, P, N, univ, tradeidx, ret):
    #P is the lookahead, in months
    universe = univ[h,:N]
    today = tradeidx[h][0]
    investPeriod = range(today, today + P*21)
    outRet = ret[investPeriod][:, universe]
    return outRet


# In[ ]:


def get_past_period(h, T, N, univ, tradeidx, ret):
    universe = univ[h,:N]
    today = tradeidx[h][0]
    pastPeriod = range(today-T, today)
    pastRet = ret[pastPeriod][:, universe]
    return pastRet


# In[ ]:


def retConstShare(retMat, w):
    n, p = retMat.shape
    if len(w.shape) == 1:
        w = np.expand_dims(w, 1)
    assert(w.shape == (p,1))
    wSum1 = w/np.sum(w)

    totalRetMat = 1 + retMat

    cummProdd = np.cumprod(totalRetMat, axis = 0)
    navVec = np.matmul(cummProdd, wSum1)

    wEnd = cummProdd[n-1, :]
    wEnd = np.dot(wEnd, w) #since w is (p,1) but wEnd is (1,p)
    wEnd = wEnd/np.sum(wEnd)
    wEnd = wEnd.T

    navVecTot = np.concatenate((np.ones((1,1)), navVec[:(n-1),]))

    totalRetVec = np.divide(navVec, navVecTot)

    retVec = totalRetVec - 1
    retVec = retVec * np.sum(w)

    return np.sum(retVec) #sum of all of the returns


# In[ ]:


def optimal_weights(cov):
    n = cov.shape[0]
    prec = np.linalg.inv(cov)
    denom = np.matmul(np.matmul(np.ones(n), prec), np.ones(n))
    return np.matmul(prec, np.ones(n)) / denom


# In[ ]:


def get_IR(rets):   
    avg = 100 * 12 * np.mean(rets)
    std = 100 * np.sqrt(12)*float(np.std(rets))
    if std == 0:
        return 0, 0, 0
    else:
        return avg, std, avg/std

