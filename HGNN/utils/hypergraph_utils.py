# --------------------------------------------------------
# Utility functions for Hypergraph
#定义的常用函数和类-接口
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np

# 在知道节点的特征W通过特征的距离来生成 G矩阵
def Eu_dis(x):
    """
    计算x的每个原始点之间的距离
    :param x: N X D（N*D）
                N: number
                D: 特征维度
    :return: N X N 距离矩阵
    """
    x = np.mat(x)
    # np.multiply 对这两个数组的对应元素进行计算
    # np.sum 用于矩阵的求和
    # axis = 0，就是按照列进行加，得到一个1 * n的行向量；
    # axis = 1，表示矩阵每一个行相加，得到一个m * 1的列向量。
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    # np.sqrt 用于按元素确定数组的正平方根
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

# 返回 融合特征矩阵
def feature_concat(*F_list, normal_col=False):
    """
    连接多模态特征。如果特征矩阵的维数大于2，
    该函数将其缩减为二维（使用最后一个维度作为特征维度，另一个维度将融合为对象维度）
    :param F_list: 特征矩阵列表
    :param normal_col: 规范化特征的每一列
    :return: 融合特征矩阵
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # 处理维度超过2的
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # 规范化每一列
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            # 将第一特征矩阵附加到融合特征矩阵
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))#将参数元组的元素数组按水平方向进行叠加
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)#求最值
        features = features / features_max
    return features

# 返回 融合超图关联矩阵
def hyperedge_concat(*H_list):
    """
    在H_列表中连接超边组
    :param H_list: 包含两个或多个超图关联矩阵的超边群
    :return: 融合超图关联矩阵
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

# 从关联矩阵H计算G，返回G矩阵
def generate_G_from_H(H, variable_weight=False):
    """
    从关联矩阵H计算G
    :param H: 超图关联矩阵H
    :param variable_weight: 可变权重：hyperedge的权重是否可变
    :return: G
    """
    if type(H) != list:   #返回参数的类型
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:    # 遍历H，在G数组后加上相应的元素
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    从关联矩阵H计算G
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)    #把列表中数据转换为矩阵或者向量，用于创建一个组
    n_edge = H.shape[1] #读取矩阵第二维度的长度
    # 超边缘的权重
    W = np.ones(n_edge)  #n_edge全1
    # 节点的阶数  np.sum用于矩阵的求和
    DV = np.sum(H * W, axis=1) #维度：1*4684（H的行（训练和测试数据和））
    # 超边的度
    DE = np.sum(H, axis=0)
    # np.diag()如果v为一个二维数组（矩阵），则返回该二维数组（矩阵）的K对角线元素，即用来提取返回对角线元素的作用。
    # 如果v为一个一维数组的话，则返回一个以该一维数组作为k对角线元素的矩阵，即用来生成一个对角线元素数组。
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # np.mat创建矩阵
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T#转置

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    由超图节点距离矩阵构造超图关联矩阵
    :param dis_mat: 节点距离矩阵
    :param k_neig: K近邻
    :param is_probH: prob顶点边矩阵或二进制
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # 从每个节点的中心特征空间构造超边
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    从原始节点特征矩阵初始化多尺度超图顶点边矩阵
    :param X: N_object x feature_number
    :param K_neigs: 邻居扩展的数量
    :param split_diff_scale: 是否在不同的邻域尺度上拆分超边组
    :param is_probH: prob顶点边矩阵或二进制
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
