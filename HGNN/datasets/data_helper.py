import scipy.io as scio
import numpy as np

# 读取 .mat 文件
def load_ft(data_dir, feature_name):
    data = scio.loadmat(data_dir)
    #scio.loadmat读取出来的数据是 dict 型，要先转换数据类型
    lbls = data['Y'].astype(np.long) #Y是标签。实现变量类型强制转换成long
    if lbls.min() == 1:  #lable从0开始
        lbls = lbls - 1
    # item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
    # 取出单元素张量的元素值并返回该值。矩阵可以看做是
    idx = data['indices'].item()

    if feature_name == 'AvBlock':
        fts = data['X'][0].item().astype(np.float32)
    elif feature_name == 'DCT':
        fts = data['X'][1].item().astype(np.float32)
    elif feature_name == 'DWT':
        fts = data['X'][2].item().astype(np.float32)
    elif feature_name == 'HOG':
        fts = data['X'][3].item().astype(np.float32)
    elif feature_name == 'Pse':
        fts = data['X'][4].item().astype(np.float32)

    else:
        print(f'wrong feature name{feature_name}!')
        raise IOError
    # 当where内只有一个参数时，那个参数表示条件，
    # 当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    # 为了通过indices里面1和0的数量，划分训练集与测试集
    idx_train = np.where(idx == 1)[0]
    idx_test = np.where(idx == 0)[0]
    return fts, lbls, idx_train, idx_test

