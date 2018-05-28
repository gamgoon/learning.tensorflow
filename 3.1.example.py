import numpy as np
import tensorflow as tf

# 데이터를 생성하고 결과를 시물레이션
x_data = np.random.randn(2000,3) # 3개의 특징을 가진 백터
w_real = [0.3,0.5,0.1] # 가중치
b_real = -0.2 # 편향값

# print(w_real) # (3,)
# print(x_data.shape) # (2000,3)
# print(x_data)
# print(x_data.T.shape) # (3, 2000)

noise = np.random.randn(1, 2000) * 0.1 # 가우시안 노이즈
print(noise)
y_data = np.matmul(w_real, x_data.T) + b_real + noise
# print(y_data)

NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None,3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred)) # 평균제곱오차 MSE (mean square error)

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    # 시작하기 전에 변수를 초기화한다
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        
        print(10, sess.run([w,b]))
    