# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:12:33 2020
获得pssm矩阵
@author: QYQ
"""
import numpy as np
import os
class PSSM:

    def __init__(self , db_filepath , input_filepath , output_filepath):#文件路径输入记得加r
        assert os.path.isfile(db_filepath),'数据库文件位置不存在'
        assert os.path.isfile(input_filepath),'输入文件位置不存在'
        assert os.path.isfile(output_filepath),'输出文件位置不存在'
        self.__db_filepath = db_filepath
        self.__input_filepath = input_filepath
        self.__output_filepath = output_filepath
        
        
    def getSequence(self):#序列处理
        return self.__Sequence
    def setSequence(self,Sequence):
        '''
        把序列(字符串)写入输入文件
        会自动清除fasta文件内容，然后写入，头可以忽略
        Sequence序列不区分大小写，没有要求，不符合规范的程序会自己忽略掉
        '''
        self.__Sequence = Sequence
        content = '>1P0WA|2' + '\n' + Sequence
        with open(self.__input_filepath , 'w+') as f:
            read_date = f.read()
            f.seek(0)
            f.truncate()
            f.write(content)
            f.close()
            
    def get_line_numpy( self , line ,start=11 , end=91 , step=4):
        '''
        输入一行，返回处理后的np数组
        默认start是11，end是91，step是4
        start=10 end=69 step=3
        '''
        np_line = np.zeros([20])
        index = 0
        for symbol , num in zip(line[start:end:step] , line[start+1:end:step]):
            if symbol == '-' :
                statu = -1
            else:
                statu = 1
            np_line[index] = statu*int(num)
            index = index + 1 
        return np_line   
    
    def getPSSM(self):
        with open(self.__output_filepath , 'r') as f:
            for i in range(2):#前2行无用
                f.readline()
            line = f.readline()#第三行判断
            if line[12] == 'A':
                line = f.readline()
                pssm_np = self.get_line_numpy( line )
                line = f.readline()
                while line != '\n' :
                    pssm_np = np.vstack( [pssm_np , self.get_line_numpy( line )] )
                    line = f.readline()
            else:
                line = f.readline()
                pssm_np = self.get_line_numpy( line ,10,69,3)
                line = f.readline()
                while line != '\n' :
                    pssm_np = np.vstack( [pssm_np , self.get_line_numpy( line ,10,69,3)] )
                    line = f.readline()
            f.close()
        return pssm_np  
 
    def setPSSM(self):
        '''
        生成pssm文件
        返回0 说明成功，返回1说明失败
        软件生成pssm会自动重写文件内容
        '''
        #psiblast -in_msa D:\NCBI\input\test.fasta -db D:\NCBI\blast-2.3.0+\db\uniprot-all.fasta -comp_based_stats 1 -inclusion_ethresh  0.001 -num_iterations 3 -out_ascii_pssm D:\NCBI\output\1.pssm
        cmd = 'psiblast -in_msa '+self.__input_filepath + ' -db ' + self.__db_filepath +r' -comp_based_stats 1 -inclusion_ethresh  0.001 -num_iterations 3 -out_ascii_pssm ' + self.__output_filepath
        statu = os.system(cmd)
        assert statu == 0 , '生成PSSM文件失败，cmd命令有错'

if __name__ == "__main__":
    pssm = PSSM(r'D:\NCBI\blast-2.3.0+\db\uniprot-all.fasta',r'D:\NCBI\input\test.fasta',r'D:\NCBI\output\1.pssm')
    Se = 'KKEKSPKGKSSISPQARAFLEEVFRRKQSLNSKEKEEVAKKCGITPLQVRVWFINKRMRSK'
    pssm.setSequence(Se)#写入序列
    pssm.setPSSM()#生成pssm文件
    pssm1 = pssm.getPSSM()
