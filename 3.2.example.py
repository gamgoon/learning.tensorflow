import numpy as np
import tensorflow as tf

N = 20000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 데이터를 생성하고 결과를 시물레이션
x_data = np.random.randn(N,3) # 3개의 특징을 가진 백터
w_real = [0.3,0.5,0.1] # 가중치
b_real = -0.2 # 편향값
wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1, y_data_pre_noise)

