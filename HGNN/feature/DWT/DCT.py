# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:34:50 2020

@author: Administrator
"""
import math
import numpy as np
def dct_1D( line , n):
    '''
    input:
        line：一维numpy，离散数据
        n: 取结果前n个集中的数据能量
    output:
        Re：n维矩阵
    '''
    Re = np.zeros( n )
    L = len( line )
    for u in range(n):
        if u==0 :
            Cu = (1/L)**0.5
        else:
            Cu = (2/L)**0.5
        summ = 0
        for f_i,i in zip( line , range(L) ):
            summ = summ + f_i*math.cos((i + 0.5)*math.pi*u/L)
        else:
            Re[u] = Cu*summ
    return Re

def dct_2D( pssm ):
    '''
    input:
        pssm: numpy二维矩阵
    output:
        Re:
    '''
    n = len( pssm )
    m = len( pssm[0] )
    #补齐矩阵
    if n > m:
        t = np.zeros( [n,n-m] )
        pssm = np.hstack( [pssm,t] )
    elif n<m:
        t = np.zeros( [m-n,m])
        pssm = np.vstack( [pssm,t] )
    M = max( n , m )
    A = np.zeros( [M,M] )
    for i in range(M):
        for j in range(M):
            if i==0 :
                a = math.sqrt( 1/M )
            else:
                a = math.sqrt( 2/M )
            A[i][j] = a*math.cos( math.pi*(j + 0.5)*i/M)
    Re = np.dot( A,pssm )
    Re = np.dot( Re,A.T )
    return Re[:n,:m]
    
    
    
    
    
    
    
    
    
    