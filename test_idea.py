'''
一个循环迭代中， 有a/b和 b/a， a,b都可能等于0，怎么添加平滑项进行处理
但问题就是，迭代过程需要使用 a/b和 b/a的值，然而这样处理会导致循环迭代进入死循环，即前后两轮的 a/b和b/a的值一样。

# np.allclose(eta, eta_pre, atol=1e-12) not 绝对误差大于1e-12  not np.array_equal(eta, eta_pre)
# eta_diff = [np.abs(ai - eta[i]) > 1e-50 for i, ai in enumerate(eta_pre)]
assert all(eta_diff), "diag_x 前后两轮完全一样，陷入死循环"

# assert all(xi != 0.0 for xi in E_std), "样本误差E 的标准差某一列存在为0的情况"  # assert expr, expr 为 False 时执行
# assert all(xi != 0.0 for xi in r_std), "标签误差r 的标准差存在为0的情况"
'''
