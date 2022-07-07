import xlrd as xd
import matplotlib.pyplot as plt
# 从Excel表中读取数据
def plt_Train(x,trainLoss=[],TAcc=[]):
    max_num = max(TAcc)  # 0.8
    fin_max = round(max_num, 3)
    print(fin_max)
    index_max = TAcc.index(max_num)  # 59
    print(index_max)
    # 绘制图片
    plt.xlabel('Train-Epoch')
    plt.ylabel('Acc')
    # plt.plot(TAcc, 'r-o')
    plt.plot(trainLoss, TAcc)
    show_max1 = str(fin_max)
    plt.plot(index_max * 10, fin_max, marker='o',markersize=3)
    plt.annotate(show_max1, xytext=(index_max * 10, fin_max), xy=(index_max * 10, fin_max))
    plt.savefig('E:/USERPROG/HGNN/data/plt/' + x + 'Train.png')
    plt.show()


def plt_Test(x,trainLoss=[],TAcc=[]):
    max_num = max(TAcc)  # 0.8
    fin_max = round(max_num, 3)
    index_max = TAcc.index(max_num)  # 59
    # 绘制图片
    plt.xlabel('Test-Epoch')
    plt.ylabel('Acc')
    # plt.plot(TAcc, 'r-o')
    plt.plot(trainLoss, TAcc)
    show_max1 = str(fin_max)
    plt.plot(index_max * 10, fin_max, marker='o',markersize=3)
    plt.annotate(show_max1, xytext=(index_max * 10, fin_max), xy=(index_max * 10, fin_max))
    plt.savefig('E:/USERPROG/HGNN/data/plt/' + x + 'Test.png')
    plt.show()


if __name__ == "__main__":
    x='D4'
    data =xd.open_workbook (r'E:\USERPROG\HGNN\data\plt\D4.xls') #打开excel表所在路径

    sheetT = data.sheets()[0]  #读取数据，以excel表名来打开
    dT = []
    for r in range(sheetT.ncols): #将表中数据按列逐步添加到列表中，最后转换为list结构
        dataT = []
        for c in range(sheetT.nrows):
            dataT.append(sheetT.cell_value(c,r))
        dT.append(list(dataT))
    trainLoss=dT[0]
    TAcc=dT[1]

    sheetV = data.sheets()[1]  #读取数据，以excel表名来打开
    dV = []
    for r in range(sheetV.ncols): #将表中数据按列逐步添加到列表中，最后转换为list结构
        dataV = []
        for c in range(sheetV.nrows):
            dataV.append(sheetV.cell_value(c,r))
        dV.append(list(dataV))
    ValLoss=dV[0]
    VAcc=dV[1]
    print(trainLoss)
    print(TAcc)
    print(ValLoss)
    print(VAcc)
    plt_Train(x,trainLoss,TAcc)
    plt_Test(x,ValLoss, VAcc)

