from torch import nn
from models import HGNN_conv
import torch.nn.functional as F

# HGNN继承自nn.Module，通过super（python中的超类）完成父类的初始化。
class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)#卷积层   2710，128
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))#激活函数，负的变0
        # 当training是真的时候，才会将一部分元素置为0，其他元素会乘以scale1 / (1 - p).training
        # 为false的时候，dropout不起作用
        x = F.dropout(x, self.dropout)#默认training=True
        x = self.hgc2(x, G)
        return x
