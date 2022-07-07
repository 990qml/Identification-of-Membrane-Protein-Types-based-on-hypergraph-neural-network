# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:42:31 2019
二维DCT变换，只找到一维DCT  cv2.dct()
应该可以通过矩阵来变换，但是不会，通过多层for循环太慢了
这里取的是左上角（三角形，之字形）的矩阵
@author: QYQ
"""
import numpy as np
import math        
def get_DCT( pssm ):
    '''
    Parameters
    ----------
    pssm : numpy
        论文公式.
    Returns
    -------
    PSSM_DCT 100维的一维数组.
    '''
    N = len(pssm)#n行m列
    M = len(pssm[0])
    DCT = np.zeros( [N,M] )
    for i in range(N):
        if i == 0:
            ai = math.sqrt(1/N)
        else:
            ai = math.sqrt(2/N)
        for j in range(M):
            if j == 0:
                aj = math.sqrt(1/M)
            else:
                aj = math.sqrt(2/M)
            DCT[i,j] = ai * aj * Mat( pssm , i , j )
    return DCT    

def get_100( DCT ):
    '''
    压缩后，包含PSSM较多信息的低频部分分布在压缩后的PSSM矩阵的左上角,取左上角前100个
    ----------
    DCT : numpy
        压缩后的信息.
    Returns
    -------
    pssm_DCT : numpy
        1*100维.
    '''
    pssm_DCT=np.zeros(100)
    Count=0
    pssm_DCT[0]=DCT[0][0]
    for i in range(1,21):
        if i%2==1:
            for j in range(i+1):
                Count+=1
                pssm_DCT[Count]=DCT[j][i-j]
                if Count==99:
                    break
                if i-j==0:
                    break
        if i%2==0:
            for j in range(i+1):
                Count+=1
                pssm_DCT[Count]=DCT[i-j][j]
                if Count==99:
                    break
                if i-j==0:
                    break
        if Count==99:
            break
    return pssm_DCT
def Mat( pssm , i , j ):
    '''
    Mat
    ----------
    pssm : numpy
        DESCRIPTION.
    i : index
        DESCRIPTION.
    j : index
        DESCRIPTION.
    Returns
    -------
    count : int
        总和.
    '''
    N = len(pssm)#n行m列
    M = len(pssm[0])
    count = 0
    for n in range(N):
        for m in range(M):
            count = count + pssm[n,m] * math.cos(math.pi*(2*n+1)*i/(2*N)) * math.cos(math.pi*(2*m+1)*j/(2*M))
    else:
        return count

        