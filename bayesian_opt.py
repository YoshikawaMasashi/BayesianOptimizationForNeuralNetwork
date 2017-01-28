#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:09:10 2017

@author: marshi
"""

import GPyOpt
import numpy as np 
import mnist_nn

def f(x):
    '''
    mnist_nn.test_mnist_nnのラッパー関数
    _x[0](2~3.5) -> epoch_num = int(10**_x[0])
    _x[1](1~2.5) -> batch_size = int(10**_x[1])
    _x[2](0~0.9) -> dropout_r = np.float32(_x[2])
    _x[3](0.51~5) -> h_cnl = np.float32(_x[3])
    _x[4](0,1) -> conv = bool(_x[4])
    _x[5](1,2,3,4) -> layers = int(_x[5])
    _x[6](0,1) -> residual = bool(_x[6])
    _x[7](5,7,9,11,13) -> conv_ksize = int(_x[7])
    '''
    ret = []
    for _x in x:
        print(_x)
        _ret = mnist_nn.test_mnist_nn(epoch_num = int(10**_x[0]),
                                        batch_size = int(10**_x[1]),
                                        dropout_r = np.float32(_x[2]),
                                        h_cnl = np.float32(_x[3]),
                                        conv = bool(_x[4]),
                                        layers = int(_x[5]),
                                        residual = bool(_x[6]),
                                        conv_ksize = int(_x[7]))
        ret.append(_ret)
    ret = np.array(ret)
    return ret

#それぞれの変数の領域を指定
bounds = [{'name': 'log10_epochs', 'type': 'continuous', 'domain': (2,3.5)},
          {'name': 'log10_batch_size', 'type': 'continuous', 'domain': (1,2.5)},
          {'name': 'dropout_r', 'type': 'continuous', 'domain': (0.0,0.9)},
          {'name': 'h_cnl', 'type': 'continuous', 'domain': (0.51,5)},
          {'name': 'conv', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'residual', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'conv_ksize', 'type': 'discrete', 'domain': (5,7,9,11,13)}]

#ベイズ最適化オブジェクト作成，最初のランダムサンプリング(20回)
myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=20)

#ベイズ最適化100ステップとステップごとの結果の表示
for i in range(100):
    myBopt.run_optimization(max_iter=1)
    print(i,myBopt.fx_opt)
    print(myBopt.x_opt)