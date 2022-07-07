# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:11:41 2020
k折交叉检验
@author: qyq
"""
import numpy as np
import sys
sys.path.append(r'D:\libsvm-3.24\python')
from svm import*
from svmutil import*
def k_cross( arg1 , arg2 ,train_x , train_y , cv):
    '''
    input:
        arg1:参数1，数列
        arg2:参数2，数列
        train_x：训练集，原始训练集  不是自定义核不需要加索引
        train_y：标签，默认数列 或者 1_Dnp
        cv：交叉检验 5 
    '''
    acc_list = []
    for i in arg1:
        for j in arg2:
            acc = 0
            for n in range(cv):
                i_train_x,i_train_y,i_test_x,i_test_y = divide_x_y(train_x , train_y , cv , n)
                option = '-t 2 -c '+str(i)+' -g '+str(j)
                model = svm_train(i_train_y , i_train_x , option)
                y_pre , ACC , y_score = svm_predict( i_test_y , i_test_x , model)
                acc = acc + ACC[0]
            acc_list.append( acc/cv )
    acc_list = np.array( acc_list )
    acc_list = acc_list.reshape( [len(arg1) , len(arg2)] )
    return acc_list

def k_cross_1( arg1  ,train_x , train_y , cv):
    '''
    input:
        arg1:参数1，数列
        arg2:参数2，数列
        train_x：训练集，原始训练集  不是自定义核不需要加索引
        train_y：标签，默认数列 或者 1_Dnp
        cv：交叉检验 5 
    '''
    acc_list = []
    for i in arg1:
        acc = 0
        for n in range(cv):
            i_train_x,i_train_y,i_test_x,i_test_y = divide_x_y(train_x , train_y , cv , n)
            option = '-t 0 -c '+str(i)
            model = svm_train(i_train_y , i_train_x , option)
            y_pre , ACC , y_score = svm_predict( i_test_y , i_test_x , model)
            acc = acc + ACC[0]
        acc_list.append( acc/cv )
    acc_list = np.array( acc_list )
    return acc_list

def add_index(np_a):#增加索引
    '''
    input: np数组
    output: 带有索引的np数组
    '''
    index_list = [i for i in range(1,len(np_a)+1)]
    index_np = np.array(index_list)
    index_np.resize(len(index_np),1)
    return np.hstack((index_np,np_a))

def divide_x_y(x,y,cv,n):
    '''
    将训练集均匀的分开
    input:
        x 特征向量
        y label
        cv 几折，也是step
        n 几折在cv的范围内
    output：
        train_x
        test_x
        train_y 和原始格式一样
        test_y
    '''
    l = len(x)
    test_index = [j for j in range(n,l,cv)]
    all_index = [j for j in range(l)]
    train_index = list(set(all_index).difference(set(test_index)))#加了一个listQAQ,set返回的不是一个list
         
         
    test_x = x[test_index]
    test_y = y[test_index]
    train_x = x[train_index]     
    train_y = y[train_index]
    return train_x,train_y,test_x,test_y

if __name__ =="__main__":
    train = np.load(r'D:\DNAsite\RNA_N6_dataset\M41_DNC.npy')
    label = np.load(r'D:\DNAsite\RNA_N6_dataset\M41_label.npy')
    label = label.flatten()
    arg1 = [2**-5,2**-4,2**-3,2**-2,2**-1,1,2,4,8,16,32]
    arg2 = arg1
    acc_list = k_cross( arg1 , arg2 ,train , label , 5)