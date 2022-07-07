import xlrd as xd
import matplotlib.pyplot as plt
# 从Excel表中读取数据
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

# 绘制图片
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(trainLoss,TAcc,label= "train")
#plt.plot(trainLoss,VAcc)
# plt.savefig('E:/USERPROG/HGNN/data/plt/'+x+'Train.png')
# plt.show()

# plt.xlabel('Test-Epoch')
# plt.ylabel('Acc')
plt.plot(ValLoss,VAcc,label= "test")
# plt.savefig('E:/USERPROG/HGNN/data/plt/'+x+'Test.png')
plt.legend(loc="lower right")
plt.show()