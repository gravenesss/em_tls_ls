""" 绘制结果曲线 """
import os
import numpy as np
import matplotlib.pyplot as plt
from util.feature_select import getXyByStr_fn

# notok： 得提供数据进行测试 plot_xys, plot_x_wbs
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 绘制 x 和各种 y 的曲线
def plotXYs_fn(x, ys, x_label, y_label, labels, markers, file_dir, file_name, title='RMSE', need_save=True):
    # 绘制每一个 y
    for i in range(len(ys)):
        plt.plot(x, ys[i], label=labels[i], marker=markers[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # 进行保存和显示
    if need_save:
        plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


# 绘制 x 和 wb 的曲线。 x为噪声比例增大 或 训练集比例增大。 ys是参数wb的结合。
def plotXWbs_fn(sequence, ys, x_label, labels, markers, feature_len, file_dir, file_name, need_save=True):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # 遍历每个子图 0~6, 最后一个是b
    for i, ax in enumerate(axes.flatten()):
        y_label = 'b' if i == feature_len else 'w' + str(i + 1)

        for j, y in enumerate(ys):
            ax.plot(sequence, y[:, i], label=labels[j], marker=markers[j])

        ax.set_xlabel(x_label)
        ax.set_ylabel('The value of ' + y_label)
        # ax.set_title(title)   # 设置子图标题
        ax.legend()  # 添加图例
    plt.tight_layout()  # 调整子图布局的间距

    # 进行保存和显示
    if need_save:
        plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


# 绘制 观测值和预测值 的关系  todo: 如何给出多组进行绘制
def plotObservePredict_fn(observe_tls, predict_tls, observe_em, predict_em, file_dir, file_name, need_save=True):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))  # sharey=True 设置子图共享 y 轴的参数.
    # 定义数据集、颜色和标签
    datasets = {"tls": {"observe_tls": observe_tls, "predict_tls": predict_tls},
                "em": {"observe_em": observe_em, "predict_em": predict_em}}
    colors = ['blue', 'red', 'orange']
    labels = ["data1", "data2", "data3"]

    for ax, (model, data_sets) in zip(axes, datasets.items()):
        # (model, data_sets)  "tls": {"observe_tls": observe_tls, "predict_tls": predict_tls
        observe = data_sets['observe_'+model]
        predict = data_sets['predict_'+model]

        residues = []
        for observed, predicted, color, label in zip(observe, predict, colors, labels):
            # alpha指定散点的透明度，范围为 [0, 1]。  单独设置：ax.set_xlim(x_min, x_max)
            ax.scatter(observed, predicted, color=color, label=label, alpha=0.7)

            # print(np.array(observed).shape, np.array(predicted).shape)
            # print(np.array(observed - predicted).shape)
            residues.append(observed - predicted)

        # 残差直方图(预测-观察)在主要及次要测试数据。 异常值的残差未绘制在图中
        ax_ins = inset_axes(ax, width="40%", height="30%", loc="lower right", bbox_to_anchor=(0., 0., 1, 1),
                            bbox_transform=ax.transAxes, borderpad=2)
        ax_ins.set_xlim(-500, 500)
        residues = np.concatenate(residues)  # 结合为 124*1 的矩阵
        # print(np.array(residues).shape)
        ax_ins.hist(residues, bins=10, color='black', alpha=0.6)

        # 设置标题和标签
        ax.set_title(model)
        ax.set_xlabel('Observed Value')
        ax.set_ylabel('Predicted Value')
        ax.plot([0, 2300], [0, 2300], 'k--')  # 添加对角线

    # 添加整体的legend: ncol: number of columns 设置图例分为n列展示; loc: 图例所有figure位置。
    # handles：此处 有两个ax对象，取第一个；然后调用方法返回 句柄列表 和 标签列表，只需要句柄。 不加该参数，会绘制两行说明
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], loc="lower center", ncol=3)

    # 进行保存和显示
    if need_save:
        plt.savefig(os.path.join(file_dir, file_name))
    plt.show()


# 测试绘制 观测值和预测值  ok
def testObservePredict_fn():
    # 获取数据  未取对数
    select_feature = ['F2', 'F3', 'F5', 'F6', 'F9']
    data_x, data_y = getXyByStr_fn(select_feature)
    # 数据集划分
    data_x1, data_y1 = data_x[:41], data_y[:41]
    data_x2, data_y2 = data_x[41:84], data_y[41:84]
    data_x3, data_y3 = data_x[84:], data_y[84:]
    tls01_wb = [-1005.3970306519396, -763897.308768216, 8213.917221459584, 20.243474087151572, 16735.039908469444,
                -12089.633851719325]
    em01_wb = [-808.5105463897994, -293146.5123809276, 3211.4378371391012, 81.44783807780382, 15151.648763214771,
               -6595.785372501234]
    tls_p1 = (np.dot(data_x1, tls01_wb[0:5]) + tls01_wb[5]).reshape(-1, 1)
    tls_p2 = (np.dot(data_x2, tls01_wb[0:5]) + tls01_wb[5]).reshape(-1, 1)
    tls_p3 = (np.dot(data_x3, tls01_wb[0:5]) + tls01_wb[5]).reshape(-1, 1)
    em_p1 = (np.dot(data_x1, em01_wb[0:5]) + em01_wb[5]).reshape(-1, 1)
    em_p2 = (np.dot(data_x2, em01_wb[0:5]) + em01_wb[5]).reshape(-1, 1)
    em_p3 = (np.dot(data_x3, em01_wb[0:5]) + em01_wb[5]).reshape(-1, 1)
    observe_tls = [data_y1, data_y2, data_y3]
    observe_em = [data_y1, data_y2, data_y3]
    predict_tls = [tls_p1, tls_p2, tls_p3]
    predict_em = [em_p1, em_p2, em_p3]
    print(len(observe_tls),  len(predict_tls))

    plotObservePredict_fn(observe_tls, predict_tls, observe_em, predict_em, 'o_p_images', 'o_p_test.png')

    # y取对数的
    # tls01_wb = [-0.495900640555514, 48.451680744530876, 3.7752040615656237, -0.012488252692644088, -19.058494986555452,
    #             -2.9531615569744947]
    # em01_wb = [-0.4138607092608072, 88.396515823067, 2.9367191243962614, 0.005299963978443411, -0.18788603891936156,
    #            -1.9231505549961052]


if __name__ == '__main__':
    testObservePredict_fn()

    pass

'''
marker 取值及其对应的标记样式说明：
'.'：小圆点
','：像素点
'o'：实心圆
'v'：倒三角
'^'：正三角
'<'：左箭头
'>'：右箭头
'1'：下箭头
'2'：上箭头
's'：实心方块
'p'：五边形
'*'：星号
'h'：六边形1
'H'：六边形2
'+'：加号
'x'：乘号
'D'：菱形
'd'：小菱形


RuntimeWarning: Glyph 8722 missing from current font. font.set_text(s, 0.0, flags=flags)
利用matplotlib绘图时负号不显示，且报错8722
解决办法：plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
'''
