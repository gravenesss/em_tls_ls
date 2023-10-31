import os
import matplotlib.pyplot as plt
import pandas as pd
import csv


# ok：均已经给数据进行测试
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号。画图之前调用
plt.rcParams['font.family'] = ['SimSun']


# 保存 DataFrame 到 file_dir/file_name
def save_df(comments, header, df, file_dir, file_name):
    # 获取当前时间作为文件名
    filename = os.path.join(file_dir, file_name)
    # 保存到 csv 文件
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        # 使用 utf8编码，并添加了-sig以确保在Excel中正确显示中文字符
        writer = csv.writer(file)
        # 写入注解
        writer.writerow(['##COMMENT_START##'])
        for comment in comments:
            writer.writerow([comment])
        writer.writerow(['##COMMENT_END##'])
        writer.writerow(header)
        writer.writerows(df.values)


# 保存 x, y... x_labels, y_labels 到 file_dir/file_name  ---- y 以行 的形式保存。
def save_csv_row(x, ys, x_labels, y_labels, comments, file_dir, file_name):
    data_all = [x] + ys
    header = [x_labels] + y_labels
    df = pd.DataFrame(data_all, header)

    save_df(comments, header, df, file_dir, file_name)


# 保存 x, y... x_labels, y_labels 到 file_dir/file_name  ---- y 以列 的形式保存。
def save_csv_col(x, ys, x_labels, y_labels, comments, file_dir, file_name):
    header = [x_labels] + y_labels
    df = pd.DataFrame()
    df[x_labels] = x
    for i, label in enumerate(y_labels):
        df[label] = ys[i]

    save_df(comments, header, df, file_dir, file_name)


# 以行/列的方式读取文件
def get_df(filename, row_or_col):
    # 报错：libiomp5md.dll
    # https://blog.csdn.net/peacefairy/article/details/110528012
    # 0.读取 CSV 文件  df = pd.read_csv(filename) # 写/读方式要一致
    with open(filename, 'r', encoding='utf-8-sig') as file:
        data = file.readlines()

    # 1.解析备注信息和数据内容
    # comments_start = data.index('##COMMENT_START##\n')
    comments_end = data.index('##COMMENT_END##\n')
    # comments = [line.strip() for line in data[comments_start + 1:comments_end]]

    # 2.获取列的信息
    header = data[comments_end + 1].strip().split(',')
    # 获取具体的数据
    content = [line.strip().split(',') for line in data[comments_end + 2:]]

    # 3.创建 DataFrame Convert content list to DataFrame
    if row_or_col == 'row':
        df = pd.DataFrame(list(map(list, zip(*content))), columns=header)
        """list(map(list, zip(*content)))：将content列表进行①转置 并②将其元素转换为内部列表的操作。
        1）zip(*content)：使用*展开content中的元素，并将它们传递给zip()函数，使zip()函数对每个原始列表的相应元素进行打包。
        例如，对于原始列表content，zip()函数会将它们打包成一个新的可迭代序列，其中每个项目都是一个元组，例如：
        (0.2,0.3876467639270337,0.19867801324210893)，(0.25,0.3490801368178834  0.17945301032886535)，等等。
        2）map(list, ...)：对于上一步中生成的每个元组，将其转换为一个列表。这里使用map()函数将list()函数应用于生成的元组，从而将元组转换为列表。
        """
    else:  # if row_or_col == 'col':
        df = pd.DataFrame(content, columns=header)

    return header, df


# 读取保存的csv文件中的 rmse 进行绘制， 按行读取 todo: 可以附加绘制 wb 的情况
def plot_csv_row(filename):
    header, df = get_df(filename, 'row')

    x = df[header[0]].astype(float)
    y1 = df['tls_rmse'].astype(float)
    y2 = df['em_rmse'].astype(float)
    # 2、绘制折线图 在折点处添加圆圈。折的标记：https://matplotlib.org/stable/api/markers_api.html
    plt.plot(x, y1, label='tls', marker='o')
    plt.plot(x, y2, label='em&tls', marker='^')
    # 添加标题和标签
    plt.xlabel('Proportion of Training Data')
    plt.ylabel('Test RMSE')
    # 添加图例
    plt.legend()
    plt.show()
    pass


# 读取保存的csv文件中的 rmse 进行绘制， 按列读取
def plot_csv_col(filename):
    header, df = get_df(filename, 'col')

    x = df[header[0]].astype(float)
    y1 = df['tls_rmse'].astype(float)
    y2 = df['em_rmse'].astype(float)
    plt.plot(x, y1, label='tls', marker='o')
    plt.plot(x, y2, label='em&tls', marker='^')
    # 添加标题和标签
    plt.xlabel('Proportion of Training Data')
    plt.ylabel('Test RMSE')
    # 添加图例
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    # 1 写为行的形式； 2 以行的形式读出。  3 以列的形式写入 4 以列的形式读出。
    save_or_read = 4

    # 准备数据
    now_comment = ['训练集比例增大',
                   '使用的数据为：F2 F3 F5 F6 F9',
                   '前后两次 w 的差距：1e-6',
                   '噪声模式为：[测试保存]'
                   '噪声比例：0.2 => 0.9 (步长0.05)',
                   '随机划分数据集次数：100',
                   '随机生成噪声次数：100']
    sequence = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    tls = [0.3876467639270337, 0.3490801368178834, 0.2929356873686641, 0.2561889404502434, 0.25366105161447383,
           0.21867789591936665, 0.21878395407547196, 0.207509891097104, 0.18248199795320802, 0.17476602819870285,
           0.17350623473424187, 0.17073918767523058, 0.17144587543209344, 0.16180962212532388, 0.14725045391530728]
    em = [0.19867801324210893, 0.17945301032886535, 0.15773748683812966, 0.14975912295745547, 0.1464498630344859,
          0.13951489954822716, 0.1390783838671, 0.1363065287744576, 0.13033812007622733, 0.12836308766441845,
          0.12941284905072165, 0.1282258634764192, 0.12825218670420238, 0.1261821833333604, 0.1200409820328712]

    if save_or_read == 1:
        save_csv_row(sequence, [tls, em], 'train_ratio', ['tls_rmse', 'em_rmse'], now_comment, 'save_csv', "train1.csv")
        pass
    elif save_or_read == 2:
        plot_csv_row('save_csv/train1.csv')
        pass
    elif save_or_read == 3:
        save_csv_col(sequence, [tls, em], 'train_ratio', ['tls_rmse', 'em_rmse'], now_comment, 'save_csv', "train3.csv")
        pass
    elif save_or_read == 4:
        plot_csv_col('save_csv/train3.csv')
        pass

    pass
