import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码
# da1_Train=[610,312,24,44,1316,151,182,610]
# da1_Test=[444,78,6,12,3265,38,46,444]
labels = ['Train', 'Test']
type1 = [435,478]
type2 = [152,18]
type3 = [0,0]
type4 = [0,0]
type5 = [1311,1867]
type6= [51,14]
type7= [110,86]
type8= [0,0]
x = np.arange(len(labels))  # 标签位置
width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

fig, ax = plt.subplots()
rects1 = ax.bar(x - width*2, type1, width, label='type 1')
rects2 = ax.bar(x - width+0.01, type2, width, label='type 2')
rects3 = ax.bar(x + 0.02, type3, width, label='type 3')
rects4 = ax.bar(x + width+ 0.03, type4, width, label='type 4')
rects5 = ax.bar(x + width*2 + 0.04, type5, width, label='type 5')
rects6 = ax.bar(x + width*3 + 0.05, type6, width, label='type 6')

rects7 = ax.bar(x + width*4 + 0.06, type7, width, label='type 7')

rects8 = ax.bar(x + width*5 + 0.07, type8, width, label='type 8')



# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('膜蛋白数量', fontsize=16)
ax.set_xlabel('膜蛋白种类', fontsize=16)
ax.set_title('DATASET 4')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
autolabel(rects7)
autolabel(rects8)


fig.tight_layout()

plt.show()
