## Hypergraph Neural Networks
Created by Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong, Ji, Yue Gao from Xiamen University and Tsinghua University.

由来自厦门大学和清华大学的冯逸凡、尤浩轩、张子昭、纪荣荣、高悦共同创作

![pipline](doc/pipline.png)

### Introduction
This work will appear in AAAI 2019. We proposed a novel framework(HGNN) for data representation learning, which could take multi-modal data and exhibit superior performance gain compared with single modal or graph-based multi-modal methods. You can also check our [paper](http://gaoyue.org/paper/HGNN.pdf) for a deeper introduction.

HGNN could encode high-order data correlation in a hypergraph structure. Confronting the challenges of learning representation for complex data in real practice, we propose to incorporate such data structure in a hypergraph, which is more flexible on data modeling, especially when dealing with complex data. In this method, a hyperedge convolution operation is designed to handle the data correlation during representation learning. In this way, traditional hypergraph learning procedure can be conducted using hyperedge convolution operations efficiently. HGNN is able to learn the hidden layer representation considering the high-order data structure, which is a general framework considering the complex data correlations.

In this repository, we release code and data for train a Hypergrpah Nerual Networks for node classification on ModelNet40 dataset and NTU2012 dataset. The visual objects' feature is extracted by [MVCNN(Su et al.)](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf) and [GVCNN(Feng et al.)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf).

这项工作将发表在2019年AAAI杂志上。我们提出了一种用于数据表示学习的新框架(HGNN)，与单模态或基于图的多模态方法相比，该框架可以获取多模态数据，并显示出优越的性能增益。你也可以查看我们的[论文](http://gaoyue.org/paper/HGNN.pdf)获得更深入的介绍。

HGNN可以在超图结构中编码高阶数据相关。面对在实际应用中学习复杂数据表示的挑战，我们提出将这种数据结构合并到一个超图中，它在数据建模方面更加灵活，特别是在处理复杂数据时。在该方法中，设计了一个超边卷积运算来处理表示学习过程中的数据相关性。这样，传统的超图学习过程可以有效地利用超边卷积进行。HGNN能够学习考虑高阶数据结构的隐层表示，是一种考虑复杂数据相关性的通用框架。

在这个存储库中，我们发布了在ModelNet40数据集和NTU2012数据集上训练Hypergrpah神经网络进行节点分类的代码和数据。视觉对象的特征由[MVCNN(Su et al.)](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf)和[GVCNN(Feng et al.)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf)提取。


### Citation
if you find our work useful in your research, please consider citing:

    @article{feng2018hypergraph,
      title={Hypergraph Neural Networks},
      author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
      journal={AAAI 2019},
      year={2018}
    }

### Installation
Install [Pytorch 0.4.0](https://pytorch.org/). You also need to install yaml. The code has been tested with Python 3.6, Pytorch 0.4.0 and CUDA 9.0 on Ubuntu 16.04.

安装(Pytorch 0.4.0) (https://pytorch.org/)。您还需要安装yaml。代码已经在Python 3.6, Pytorch 0.4.0和CUDA 9.0 Ubuntu 16.04上进行了测试。

### Usage

**Firstly, you should download the feature files of modelnet40 and ntu2012 datasets.
Then, configure the "data_root" and "result_root" path in config/config.yaml.**

Download datasets for training/evaluation  (should be placed under "data_root")
- [ModelNet40_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1euw3bygLzRQm_dYj1FoRduXvsRRUG2Gr/view?usp=sharing)
- [NTU2012_mvcnn_gvcnn_feature](https://drive.google.com/file/d/1Vx4K15bW3__JPRV0KUoDWtQX8sB-vbO5/view?usp=sharing)

GVCNN和MVCNN一样，也是一种基于多视图（multi-view）的、对三维物体进行识别分类的网络结构。
在MVCNN中，各个view的CNN特征通过一个view pooling层被整合成一个特征向量。这么做的缺憾在于，
view pooling层并不能关注每个view的区分性。而实际上，在使用multi-view来对一个三维物体进行表示时，
有些view之间的相似性可能会很高，而有些view之间的差别却比较大，
挖掘出这些view之间的相互联系也许能够帮助网络更好地进行物体识别。
而GVCNN的核心思路便是基于上述的观测，它提出了一个c，用于对不同的view-level特征进行分组，
并以组为单位对view-level特征进行聚合从而得到group-level的特征，
最后通过一个学习到的权重来将group-level特征整合成一个全局特征描述子，以用于最终的分类。
引入Grouping Module的好处在于，它考虑了view之间的组内相似性和组间区分性：相似度高的view被分到了同一组，
组内特征对最终结果的影响因子是相同的；而不同的组间具有相对明显的区分性，因此每个组对最终结果的影响程度会不同。

To train and evaluate HGNN for node classification:

**首先，你应该下载modelnet40和ntu2012数据集的特征文件。

然后在config/config.yaml.**中配置“data_root”和“result_root”路径

下载用于培训/评估的数据集(应放在“data_root”下)

——(ModelNet40_mvcnn_gvcnn_feature) (https://drive.google.com/file/d/1euw3bygLzRQm_dYj1FoRduXvsRRUG2Gr/view?usp=sharing)

——(NTU2012_mvcnn_gvcnn_feature) (https://drive.google.com/file/d/1Vx4K15bW3__JPRV0KUoDWtQX8sB-vbO5/view?usp=sharing)

训练和评估HGNN进行节点分类:

```
python train.py
```
You can select the feature that contribute to construct hypregraph incidence matrix by changing the status of parameters "use_mvcnn_feature_for_structure" and "use_gvcnn_feature_for_structure" in config.yaml file. Similarly, changing the status of parameter "use_gvcnn_feature" and "use_gvcnn_feature" can control the feature HGNN feed, and both true will concatenate the mvcnn feature and gvcnn feature as the node feature in HGNN.

通过改变config中的参数“use_mvcnn_feature_for_structure”和“use_gvcnn_feature_for_structure”的状态，可以选择有助于构造超预图关联矩阵的特性。yaml文件。同样，改变参数“use_gvcnn_feature”和“use_gvcnn_feature”的状态可以控制feature HGNN feed，两者都为true会将mvcnn feature和gvcnn feature连接起来作为HGNN中的节点feature。

```yaml
# config/config.yaml
use_mvcnn_feature_for_structure: True
use_gvcnn_feature_for_structure: True
use_mvcnn_feature: False
use_gvcnn_feature: True
```
To change the experimental dataset (ModelNet40 or NTU2012)

更改实验数据集(ModelNet40或NTU2012)

```yaml
# config/config.yaml
#Model
on_dataset: &o_d ModelNet40
#on_dataset: &o_d NTU2012
```
### License
Our code is released under MIT License (see LICENSE file for details).

我们的代码是在MIT许可下发布的(详情请参阅许可文件)。

