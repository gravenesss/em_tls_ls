from utils import *


def get_rmse_Loss_by_wb1(x_test, y_test, w, b=0.0, log_base=1):
    y_predict = np.dot(x_test, w) + b
    # # Python 中可以使用 ** 运算符或内置函数 pow() 来实现幂运算
    if log_base == 10:  # log10(y) = y' y=10^y'
        y_predict = np.power(10, y_predict)
        y_test = np.power(10, y_test)
    elif log_base == np.e:
        y_predict = np.exp(y_predict)
        y_test = np.exp(y_test)
    else:
        pass
    mse = mean_squared_error(y_test, y_predict)
    rmse = mse ** 0.5
    # 计算 perr
    n = y_test.size
    perr = 0
    for i in range(n):
        perr += np.abs(y_test[i] - y_predict[i]) / y_test[i]
    perr = (perr * 100 / n)[0]
    return rmse, perr


# 计算三个数据集的 rmse
def test_y_log_y():
    # 获取数据
    select_feature = ['F2', 'F3', 'F5', 'F6', 'F9']  # 特征
    data_x, data_y = get_xy_by_str(select_feature)
    is_log = 10
    if is_log == 10:
        data_y = np.log10(data_y)

    # 数据集划分
    data_x1, data_y1 = data_x[:41], data_y[:41]
    data_x2, data_y2 = data_x[41:84], data_y[41:84]
    data_x3, data_y3 = data_x[84:], data_y[84:]

    # wb 使用的是训练集比例90%，噪声比例0.1的情况
    # 噪声模式为：[1.19143622 1.47466608 0.72362853 1.11948969 1.80730452 1.81332756]  第三个文件夹
    # (129.1408632536317, 18.85408131648979) (140.73950392712544, 19.32238557716365) (176.30969432160268, 11.86817080937198)
    # (123.27186326962558, 16.840051881736464) (125.97813587921273, 15.216344306270635) (188.5901574246523, 11.349466269098995)
    if is_log == 1:
        tls01_wb = [-1005.3970306519396, -763897.308768216, 8213.917221459584, 20.243474087151572, 16735.039908469444, -12089.633851719325]
        em01_wb = [-808.5105463897994, -293146.5123809276, 3211.4378371391012, 81.44783807780382, 15151.648763214771, -6595.785372501234]

        tls_rmse1 = get_rmse_Loss_by_wb(data_x1, data_y1, tls01_wb[0:5], tls01_wb[5])
        tls_rmse2 = get_rmse_Loss_by_wb(data_x2, data_y2, tls01_wb[0:5], tls01_wb[5])
        tls_rmse3 = get_rmse_Loss_by_wb(data_x3, data_y3, tls01_wb[0:5], tls01_wb[5])
        print(tls_rmse1, tls_rmse2, tls_rmse3)

        em_rmse1 = get_rmse_Loss_by_wb(data_x1, data_y1, em01_wb[0:5], em01_wb[5])
        em_rmse2 = get_rmse_Loss_by_wb(data_x2, data_y2, em01_wb[0:5], em01_wb[5])
        em_rmse3 = get_rmse_Loss_by_wb(data_x3, data_y3, em01_wb[0:5], em01_wb[5])
        print(em_rmse1, em_rmse2, em_rmse3)

    # (118.39158283093771, 12.864439217213711) (134.10150503975294, 11.930087432952496) (169.34455014807213, 11.833166156722514)
    # (92.08937360238471, 12.668331593745055) (109.56559377494159, 12.394894614100133) (186.99874482777017, 10.583970267504226)
    elif is_log == 10:
        tls01_wb = [-0.495900640555514, 48.451680744530876, 3.7752040615656237, -0.012488252692644088, -19.058494986555452, -2.9531615569744947]
        em01_wb = [-0.4138607092608072, 88.396515823067, 2.9367191243962614, 0.005299963978443411, -0.18788603891936156, -1.9231505549961052]

        tls_rmse1 = get_rmse_Loss_by_wb1(data_x1, data_y1, tls01_wb[0:5], tls01_wb[5], 10)
        tls_rmse2 = get_rmse_Loss_by_wb1(data_x2, data_y2, tls01_wb[0:5], tls01_wb[5], 10)
        tls_rmse3 = get_rmse_Loss_by_wb1(data_x3, data_y3, tls01_wb[0:5], tls01_wb[5], 10)
        print(tls_rmse1, tls_rmse2, tls_rmse3)

        em_rmse1 = get_rmse_Loss_by_wb1(data_x1, data_y1, em01_wb[0:5], em01_wb[5], 10)
        em_rmse2 = get_rmse_Loss_by_wb1(data_x2, data_y2, em01_wb[0:5], em01_wb[5], 10)
        em_rmse3 = get_rmse_Loss_by_wb1(data_x3, data_y3, em01_wb[0:5], em01_wb[5], 10)
        print(em_rmse1, em_rmse2, em_rmse3)

        pass


if __name__ == '__main__':
    test_y_log_y()
    pass


