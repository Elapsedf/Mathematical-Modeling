

from locale import normalize
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 可视化w

inputdata = pd.read_csv(r'C:\Users\Administrator\Desktop\APMCM\plot\new_data_mean.txt',encoding='gb18030')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


df = inputdata.copy()
_, ax = plt.subplots(figsize=(10, 7))  # 分辨率1200×1000
corr = df.corr(method='pearson')  # 使用皮尔逊系数计算列与列的相关性
# corr = df.corr(method='kendall')  # 肯德尔秩相关系数
# corr = df.corr(method='spearman') # 斯皮尔曼秩相关系数

# 上面三行代表了不同的计算方法，需要哪个就把其他的备注就好
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
# 在两种HUSL颜色之间制作不同的调色板。图的正负色彩范围为220、10，结果为真则返回matplotlib的colormap对象
_ = sns.heatmap(
    corr,  # 使用Pandas DataFrame数据，索引/列信息用于标记列和行
    cmap=cmap,  # 数据值到颜色空间的映射
    square=False,  # 每个单元格都是正方形
    cbar_kws={'shrink': .9},  # `fig.colorbar`的关键字参数
    ax=ax,  # 绘制图的轴
    annot=True,  # 在单元格中标注数据值
    annot_kws={'fontsize': 20})  # 热图，将矩形数据绘制为颜色编码矩阵
plt.title('Pearson Corr Result',size=20)
plt.show()
# plt.get_figure().savefig('斯皮尔曼秩相关系数热力图.png')#保留图片
print()