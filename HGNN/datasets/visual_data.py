from datasets import load_ft
from utils import hypergraph_utils as hgut

# 定义加载特征数据
def load_feature_construct_H(data_dir,
                             m_prob=1,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,
                             use_AvBlock_feature=False,
                             use_DCT_feature=False,
                             use_DWT_feature=False,
                             use_HOG_feature=False,
                             use_Pse_feature=False,
                             use_AvBlock_feature_for_structure=False,
                             use_DCT_feature_for_structure=False,
                             use_DWT_feature_for_structure=False,
                             use_HOG_feature_for_structure=False,
                             use_Pse_feature_for_structure=False
                             ):
    """

    :param data_dir: 特征数据目录
    :param m_prob: 超图关联矩阵构造中的参数
    :param K_neigs: 邻接扩展的数量（K近邻）
    :param is_probH: 概率顶点边矩阵或二进制
    :return:
    """
    # 初始特征
    if use_AvBlock_feature or use_AvBlock_feature_for_structure:
        AvBlock_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='AvBlock')
    if use_DCT_feature or use_DCT_feature_for_structure:
        DCT_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='DCT')
    if use_DWT_feature or use_DWT_feature_for_structure:
        DWT_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='DWT')
    if use_HOG_feature or use_HOG_feature_for_structure:
        HOG_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='HOG')
    if use_Pse_feature or use_Pse_feature_for_structure:
        Pse_ft, lbls, idx_train, idx_test = load_ft(data_dir, feature_name='Pse')
    if 'AvBlock_ft' not in dir() and 'DCT_ft' not in dir() and 'DWT_ft' not in dir() and 'HOG_ft' not in dir()  and 'Pse_ft' not in dir():
        raise Exception('None feature initialized')

    # 构造特征矩阵
    fts = None
    if use_AvBlock_feature:#fts 返回融合特征矩阵
        fts = hgut.feature_concat(fts, AvBlock_ft)
    if use_DCT_feature:
        fts = hgut.feature_concat(fts, DCT_ft)
    if use_DWT_feature:
        fts = hgut.feature_concat(fts, DWT_ft)
    if use_HOG_feature:
        fts = hgut.feature_concat(fts, HOG_ft)
    if use_Pse_feature:
        fts = hgut.feature_concat(fts, Pse_ft)
    if fts is None:
        raise Exception(f'None feature used for model!')

    # 构造超图关联矩阵
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    if use_AvBlock_feature_for_structure:#从 原始节点特征矩阵 初始化 多尺度超图顶点边矩阵
        tmp = hgut.construct_H_with_KNN(AvBlock_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp) #返回 融合超图关联矩阵
    if use_DCT_feature_for_structure:
        tmp = hgut.construct_H_with_KNN(DCT_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp)
    if use_DWT_feature_for_structure:#从 原始节点特征矩阵 初始化 多尺度超图顶点边矩阵
        tmp = hgut.construct_H_with_KNN(DWT_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp) #返回 融合超图关联矩阵
    if use_HOG_feature_for_structure:#从 原始节点特征矩阵 初始化 多尺度超图顶点边矩阵
        tmp = hgut.construct_H_with_KNN(HOG_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp) #返回 融合超图关联矩阵
    if use_Pse_feature_for_structure:#从 原始节点特征矩阵 初始化 多尺度超图顶点边矩阵
        tmp = hgut.construct_H_with_KNN(Pse_ft, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp) #返回 融合超图关联矩阵

    if H is None:
        raise Exception('None feature to construct hypergraph incidence matrix!')

    return fts, lbls, idx_train, idx_test, H
