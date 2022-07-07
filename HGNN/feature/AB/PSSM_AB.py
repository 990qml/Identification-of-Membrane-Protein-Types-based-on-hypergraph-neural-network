# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:06:10 2020
PSSM 按照每块20行分开，每块第一个之和求平均，以此类推，
每列有20个平均值，20列共有20*20个
Average Block 平均块
@author: QYQ
"""
import numpy as np
import math
def get_AB( pssm , block ):
    '''
    Average Block 分块求均值
    ----------
    pssm : numpy L*20
        pssm矩阵.
    block : int
        分20块还是分10块.
    Returns
    -------
    Re : 一维numpy
        返回的结果 长度是(1+2+3+4)*20.
    '''
    Re = np.zeros( 0 )
    for i in range(block):
        Re = np.hstack( [Re , AB_block(pssm,i+1) ])
    return Re
def AB_block( pssm , n_divid):
    '''
    input:
        pssm: numpy矩阵L*20
        n_divid: 分n_divid块
    output:
        Re: numpy矩阵
    '''
    L = len( pssm )
    Re = np.zeros( 0 )
    
    step = math.floor( L/n_divid )
    for i in range( n_divid ):
        RID = pssm[ i*step:(i+1)*step , : ]
        Re = np.hstack( [ Re , RID.mean( axis=0 ) ] )
    return Re
    
    
    
    
    
    
    
    
    