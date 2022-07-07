# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 08:54:54 2019
位置特异性打分矩阵 离散小波变换 PSSM-DWT
@author: QYQ
"""
import pywt#安装PyWavelets
import numpy as np
from DCT import dct_1D
def get_PSSM_DWT( pssm , n ):
    '''
    input:
        pssm L*20 numpy
        n int dwt次数
    output:
        Re 1040 一维矩阵
    '''
    Re = np.zeros(0)
    for l in range( len( pssm[0] ) ):
        col = pssm[:,l]
        ca = col
        for i in range(n):#循环n次
            eight = np.zeros(8)
            ca , cd = pywt.dwt( ca , 'bior3.3')
            eight[0] = ca.min()
            eight[1] = ca.max()
            eight[2] = ca.mean()
            eight[3] = ca.std( ddof=1 ) #默认ddof=0对样本进行求偏差 /（n-1） 这里改成ddof=1 对总体求偏差/n
            eight[4] = cd.min()
            eight[5] = cd.max()
            eight[6] = cd.mean()
            eight[7] = cd.std( ddof=1 )
            cc = dct_1D( ca , 5 )
            Re = np.hstack( [Re , cc ] )
            Re = np.hstack( [Re , eight ])
    else:
        return Re
            
def Min_Max_ed( dwt ):
    '''
    线性归一化      
    '''           
    Max = dwt.max( axis = 0 )
    Min = dwt.min( axis = 0 )
    return ( dwt - Min )/( Max - Min )      
            
            