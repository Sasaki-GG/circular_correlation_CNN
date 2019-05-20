#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import matplotlib.pyplot as plt

class singleConv(nn.Module):
    def __init__(self, kernel, channel):
        super(singleConv, self).__init__()
        #传入自定义的核
        self.kk = kernel
        kernel = torch.FloatTensor(kernel).unsqueeze(2)
        # print (self.kk)
        # print (kernel.shape)
        #设置参数与计算梯度
        # kernel = kernel.reshape([1, 5, 5])
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.channel = channel

    def forward(self, x):
         # 自定义卷积
        x = nn.functional.conv1d(input=x, weight=self.weight, stride=1, bias=None)
        # x = self.conv2(x)
        #一层的卷积
        return x

def calc_grad(x, kernel, y, dim, reverse=False):
    # 转换为tensor
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    kernel = torch.from_numpy(kernel).float()
    x = x.reshape([1, 1, dim])
    y = y.reshape([1, 1, dim])
    kernel = kernel.reshape([1, dim])
    #级联
    t = torch.cat((x, x), 2)
    #自定义核
    # tmp = kernel
    # for i in range(dim-1):
    #     kernel = torch.cat((kernel, tmp), 0)
    model = singleConv(kernel, dim)

    #循环左/右移
    for i in range(dim-1):
        if reverse==False:
            tmp = t[0, 0, i+1:i+dim+1]
        else:
            tmp = t[0, 0, dim-1-i:2*dim-1-i]
        tmp = tmp.reshape([1, 1, dim])
        x = torch.cat((x, tmp), 1)
    #优化器: adam, loss: L2 
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_function = nn.MSELoss()
    # print ('Input\n',x)
    #相当于一个epoch
    for epoch in range(1):
        out = model(x.float())
        loss = loss_function(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        # print (optimizer.param_groups)
        # # print ('out',out)
        # for name, weight in model.named_parameters():
        #     print("name:", name)
        #     print("weight:", weight)
        #     print("weight.grad:", weight.grad)
    #获得model的参数,找到梯度返回
    for name, weight in model.named_parameters():
        if name == 'weight':
            return weight.grad[0, :,0]

def double_grad(a, b, c):
    #这里假设输入的为numpy数组
    # a cir_conv b  = c
    dim = a.shape[0]
    #正反交换
    grada = calc_grad(b, a, c, dim, reverse=False)
    gradb = calc_grad(a, b, c, dim, reverse=True)
    #返回 a.grad, b.grad
    return grada.numpy(), gradb.numpy()

def cir_mul(a, b, dim=5):
    dim = a.shape[0]
    ans = np.zeros(dim)
    for i in range(dim):
        for j in range(dim):
            ans[i] += a[j] * b[(j+i) % dim]
    return ans

def mse(a, b):
    dim = a.shape[0]
    ans = 0.0
    for i in range(dim):
        ans += (a[i]-b[i])*(a[i]-b[i])
    return ans/dim

def normalize(a):
    tmp = 0
    for x in a:
       tmp += x*x
    return a/math.sqrt(tmp) 

if __name__ == "__main__":
    x = normalize(np.random.rand(100))
    w = normalize(np.random.rand(100))
    y = normalize(np.random.rand(100))
    loss_function = nn.MSELoss()
    y_pre = cir_mul(x, w, x.shape)
    yp = torch.autograd.Variable(torch.from_numpy(y_pre))
    yt = torch.autograd.Variable(torch.from_numpy(y))
    loss1 = mse(y_pre, y)
    loss = loss_function(yp, yt)
    print('loss-2', loss, loss1)
    # print ('a:{}\nb:{}\nc:{}\n'.format(x,w,y))
    for epoch in range(30):
        ans = double_grad(x, w, y)
        # print ('grad',ans)
        # print(ans[1] * 0.1)
        x = x - ans[0] * 0.1
        w = w - ans[1] * 0.1
        # w = np.add(w, ans[0] * 0.1)
        y_pre = cir_mul(x, w, x.shape)
        yp = torch.autograd.Variable(torch.from_numpy(y_pre))
        yt = torch.autograd.Variable(torch.from_numpy(y))
        losspp = mse(y_pre, y)
        loss = loss_function(yp, yt)
        # print ('x',x)
        # print ('w',w)
        # print ('y',y)
        print ('loss-2',loss, losspp)

