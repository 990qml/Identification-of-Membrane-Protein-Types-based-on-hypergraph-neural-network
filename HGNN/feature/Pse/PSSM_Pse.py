# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:50:44 2020
根据PSSM生成Pse
@author: QYQ
"""
import numpy as np
def get_Pse( pssm , lg ):
    '''
    input:
        pssm L*20
    output
        Re 220一维数组
    '''
    L = len(pssm)
    Mean = pssm.mean( axis=1 ).reshape([L,1])
    Std = pssm.std( axis=1 , ddof=1 ).reshape([L,1])
    pssm = (pssm - Mean) / Std
    pssm = np.nan_to_num( pssm )#std可能是0
    #计算20维度
    
    Ave = pssm.mean( axis=0 )#一维数组
    
    #计算20*10维度
    #0 20行数组
    Re = np.zeros(0)
    Tw = np.zeros(20)
    for lag in range(1,lg+1):#10
        for j in range(20):#20列
            for i in range(L-lag):#求和
                Tw[j] = Tw[j] + (pssm[i,j]-pssm[i+lag,j])**2
            else:
                Tw[j] = Tw[j] / (L-lag)
        else:
            Re = np.hstack( [Re,Tw] )
    
    Re = np.hstack( [Re,Ave] )
    return Re
#dic = np.load(r'D:\NCBI\test_pssm_296.npy',allow_pickle=True).item()
#num = dic[1]
#get_Pse(num)

def get_pssm_pse( dic ,lg ):
    Re = np.zeros(220)
    for (key , value) in dic.items():
        Re = np.vstack( [Re , get_Pse(value , lg)] )
    else:
        return Re


