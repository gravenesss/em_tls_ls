import numpy as np
from sklearn.metrics import mean_squared_error


# https://blog.csdn.net/guolindonggld/article/details/87856780

# 计算不同类型的误差
def getLoss_fn(y_predict, y_test, err_type='rmse'):
    # 计算误差
    err = 0.0
    if err_type == 'mae':       # MAE  (Mean Absolute Error， 平均绝对误差)
        err = np.mean(np.absolute(y_test - y_predict))
    elif err_type == 'mape':    # MAPE (Mean Absolute Percentage Error 平均相对误差)
        n = y_test.size
        for i in range(n):
            err += np.abs(y_test[i] - y_predict[i]) / y_test[i]
        err = (err / 100 * n)[0]
    else:                       # RMSE (Root Mean Square Error, 均方根误差)
        err = np.sqrt(mean_squared_error(y_test, y_predict))
    return err


# 根据x w b y获取 误差：
def getLossByWb_fn(x_test, y_test, w, b, convert_y='1', err_type='rmse', E=None, r=None):
    # if E is None and r is None:   # todo :编写给定E r 的方法
    #     y_predict = now_x @ w_std * std_y + mean_y

    y_predict = np.dot(x_test, w) + b
    #  是否对 y 进行还原:  log10(y_original) = y'   =>   y_original = 10^y'
    if convert_y == 'log10':
        # print("y_predict1:", y_predict, '\ny_test1:', y_test, '\n')
        y_predict = np.power(10, y_predict)  # ** 运算符或内置函数 pow() 来实现幂运算
        y_test = np.power(10, y_test)
        # print("y_predict2:", y_predict, '\ny_test2:', y_test, '\n\n')
    elif convert_y == 'loge':
        y_predict = np.exp(y_predict)
        y_test = np.exp(y_test)
    else:
        pass

    return getLoss_fn(y_predict, y_test, err_type)
