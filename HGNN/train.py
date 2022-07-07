import time
import copy
import xlwt
import torch
import torch.optim as optim
import pprint as pp
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#data-->plt文件夹里.xls存放的是训练过后的train acc 和test acc
#data-->protrin文件夹里.csv存放的是数据计算特征，A.csv存放的是膜蛋白的类别（选取相应.csv里面最大的那个数据作为特征类别）

# 配置环境，读取配置文件内容
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cfg = get_config('config/config.yaml')

# 初始化数据，选择需要训练的数据  #
data_dir = cfg['D4_ft'] if cfg['on_dataset'] == 'D4' \
    else cfg['D1']
# 加载特征数据
fts, lbls, idx_train, idx_test, H = \
    load_feature_construct_H(data_dir,
                             m_prob=cfg['m_prob'],
                             K_neigs=cfg['K_neigs'],
                             is_probH=cfg['is_probH'],
                             use_AvBlock_feature=cfg['use_AvBlock_feature'],
                             use_DCT_feature=cfg['use_DCT_feature'],
                             use_DWT_feature=cfg['use_DWT_feature'],
                             use_HOG_feature=cfg['use_HOG_feature'],
                             use_Pse_feature=cfg['use_Pse_feature'],
                             use_AvBlock_feature_for_structure=cfg['use_AvBlock_feature_for_structure'],
                             use_DCT_feature_for_structure=cfg['use_DCT_feature_for_structure'],
                             use_DWT_feature_for_structure=cfg['use_DWT_feature_for_structure'],
                             use_HOG_feature_for_structure=cfg['use_HOG_feature_for_structure'],
                             use_Pse_feature_for_structure=cfg['use_Pse_feature_for_structure'],
)
# 从关联矩阵H计算G，返回G矩阵
G = hgut.generate_G_from_H(H)
n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 将数据转换为设备
# torch.Tensor()生成单精度浮点类型的张量
fts = torch.Tensor(fts).to(device)
# long()向下取整
#squeeze()维度压缩。返回一个tensor（张量）
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()#从1970-01-01 00:00:00 UTC，开始到现在所经历的时间，以浮点数的'秒'来表示
    dataTrainLoss = []
    dataTrainAcc = []
    dataTestLoss = []
    dataTestAcc = []
    # 深复制.就是从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响。
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    max = 0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            # 格式化字符串，加上f以后，{“变量/表达式”},花括号里的变量和表达式就可以使用了
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # 每一个 epoch 都有一个训练和验证阶段
        # model.train():
        # 在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和dropout。
        # model.eval():
        # 测试过程中会使用model.eval()，这时神经网络会沿用batch normalization的值，并不使用drop out。
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()#step相当于计数.那么step_size指的是经过这么多次迭代，学习率改变一次。
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # 迭代数据
            optimizer.zero_grad()# 把梯度置零，也就是把loss关于weight的导数变成0.
            # 前向传递
            # 如果只是训练的话，追踪历史
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)#网络的前向传播：预测
                # softmax
                # print(outputs[0][0])
                loss = criterion(outputs[idx], lbls[idx])#然后将输出的outputs和原来导入的labels作为loss函数的输入就可以得到损失了：
                # 第一个存的是最大值，第二个存的是其对应的位置索引index。
                # 这里我们想要得到的是索引，所以后面用[1]。
                _, preds = torch.max(outputs, 1)#预测该样本属于哪个类别的信息

                # 训练时，反向传播 + 优化：根据参数调整
                if phase == 'train':
                    loss.backward()#反向传播求梯度。回传损失，过程中会计算梯度
                    optimizer.step()#根据这些梯度，更新所有参数
            # 至此，一个batch数据的训练就结束了！
            # 计算损失
            # 返回loss的值，叠加之后算出总loss，最后再除以batches的数量，取loss平均值。
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)#数据准确性


            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                # print(type(epoch_loss))
                if(phase=='train'):
                    dataTrainLoss.append(float(format(epoch, '.4f')))
                    dataTrainAcc.append(float(format(epoch_acc, '.4f')))
                if (phase == 'test'):
                    dataTestLoss.append(float(format(epoch, '.4f')))
                    dataTestAcc.append(float(format(epoch_acc, '.4f')))

                    ####
                    if(epoch_acc > max):
                        max = epoch_acc
                        out = outputs

            # 得到最好那次的模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best test Acc: {best_acc:4f}')
            print('*' * 20)

    ######
    # print(max)
    # print(out)
    x="D4"
    # 写文件
    with open("E:/USERPROG/HGNN/data/protein/"+x+".csv", "w", encoding="utf-8") as f:
        for i in range(len(out)):
            for j in range(len(out[i])):
                s = str(out[i][j])
                f.write(s[7: len(s) - 2] + ",")
            f.write("\n")
    # 找最大，确定所属类别
    with open("E:/USERPROG/HGNN/data/protein/A"+x+".csv", "w", encoding="utf-8") as f:
        for i in range(len(out)):
            max = 0
            for j in range(len(out[i])):
                if (out[i][j] > max):
                    max = out[i][j]
                    index = j
            f.write(str(index+1) + "\n")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc:4f}')

    #将数据写入plt文件夹下的.xls中
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('train Loss-Acc', cell_overwrite_ok=True)
    for i1 in range(len(dataTrainLoss)):
        sheet.write(i1, 0, dataTrainLoss[i1])
    for j1 in range(len(dataTrainAcc)):
        sheet.write(j1, 1, dataTrainAcc[j1])
    sheet = book.add_sheet('test-Loss', cell_overwrite_ok=True)
    for i2 in range(len(dataTestLoss)):
        sheet.write(i2, 0, dataTestLoss[i2])
    for j2 in range(len(dataTestAcc)):
        sheet.write(j2, 1, dataTestAcc[j2])
    savepath = r'E:/USERPROG/HGNN/data/plt/'+x+'.xls'
    book.save(savepath)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! class number: {n_class}")
    print(f"use AvBlock feature: {cfg['use_AvBlock_feature']}")
    print(f"use DCT feature: {cfg['use_DCT_feature']}")
    print(f"use DWT feature: {cfg['use_DWT_feature']}")
    print(f"use HOG feature: {cfg['use_HOG_feature']}")
    print(f"use Pse feature: {cfg['use_Pse_feature']}")
    print(f"use AvBlock feature for structure: {cfg['use_AvBlock_feature_for_structure']}")
    print(f"use DCT feature for structure: {cfg['use_DCT_feature_for_structure']}")
    print(f"use DWT feature for structure: {cfg['use_DWT_feature_for_structure']}")
    print(f"use HOG feature for structure: {cfg['use_HOG_feature_for_structure']}")
    print(f"use Pse feature for structure: {cfg['use_Pse_feature_for_structure']}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')

    model_ft = HGNN(in_ch=fts.shape[1], #2710
                    n_class=n_class,
                    n_hid=cfg['n_hid'],   #128
                    dropout=cfg['drop_out'])
    model_ft = model_ft.to(device)

    # 再继续训练所有参数，学习率调小一点。优化方法
    optimizer = optim.Adam(model_ft.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=cfg['weight_decay)
    # 表示按需求有外部设置调整学习率
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg['milestones'],
                                               gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    model_ft = train_model(model_ft, criterion, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq'])


if __name__ == '__main__':
    _main()
