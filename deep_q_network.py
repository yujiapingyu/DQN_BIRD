import tensorflow as tf
import sys
sys.path.append('game/')
import game.wrapped_flappy_bird as game
import cv2
import random
import numpy as np
from collections import deque

GAME = 'BIRD'
ACTIONS = 2
OBSERVE = 10000
EXPLORE = 30000
INITIAL_EPSILON = 0.0001
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
GAMMA = 0.99


# 初始化weights
def weights_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


# 初始化biases
def biases_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


# 卷积操作
def conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


# 池化操作
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 创建网络
def create_network():
    s = tf.placeholder(tf.float32, [None, 80, 80, 4])

    # 第一层卷积+池化，输出（None x 10 x 10 x 32）
    W_conv1 = weights_variable([8, 8, 4, 32])
    b_conv1 = biases_variable([32])
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积，输出（None x 5 x 5 x 64）
    W_conv2 = weights_variable([4, 4, 32, 64])
    b_conv2 = biases_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

    # 第三层卷积，输出（None x 5 x 5 x 64）
    W_conv3 = weights_variable([3, 3, 64, 64])
    b_conv3 = biases_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    # 第一层全连接层，输出（None x 512）
    W_fc1 = weights_variable([5 * 5 * 64, 512])
    b_fc1 = biases_variable([512])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 第一层全连接层，输出（None x 2）
    W_fc2 = weights_variable([512, ACTIONS])
    b_fc2 = biases_variable([ACTIONS])
    output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, output, h_fc1


# 训练神经网络
def train_network(s, output, h_fc1, sess):
    a = tf.placeholder(tf.float32, [None, ACTIONS])
    y = tf.placeholder(tf.float32, [None])

    # predict_y为当前状态预测结果
    predict_y = tf.reduce_sum(tf.multiply(output, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - predict_y))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game_state = game.GameState()

    memory = deque()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    # 获取初始state
    initial_frame, _, _ = game_state.frame_step(do_nothing)
    initial_frame = cv2.cvtColor(cv2.resize(initial_frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, initial_frame = cv2.threshold(initial_frame, 1, 255, cv2.THRESH_BINARY)
    state = np.stack((initial_frame, initial_frame, initial_frame, initial_frame), axis=2)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    check_point = tf.train.get_checkpoint_state('saved_networks')

    # 模型保存恢复
    if check_point and check_point.model_checkpoint_path:
        saver.restore(sess, check_point.model_checkpoint_path)
        print('Load successful.')
    else:
        print('Load failed.')

    epsilon = INITIAL_EPSILON
    t = 0

    while True:
        # 选择动作
        predict_action = output.eval(feed_dict={s: state.reshape(-1, 80, 80, 4)})[0]
        action = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            if random.random() < 0.1:
                action_index = 1
            else:
                action_index = 0
        else:
            action_index = np.argmax(predict_action)

        action[action_index] = 1

        # 传入动作，获取下一帧，奖励，和是否终止的标志
        frame, reward, terminal = game_state.frame_step(action)

        # 拼接出下一个状态state_
        frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (80, 80, 1))
        state_ = np.append(frame, state[:, :, :3], axis=2)

        # 存入记忆
        memory.append((state, action, reward, state_, terminal))
        if len(memory) > REPLAY_MEMORY:
            memory.popleft()

        # 如果观测够了，就开始训练
        if t > OBSERVE:
            batch = random.sample(memory, BATCH_SIZE)
            s_batch = [item[0] for item in batch]
            a_batch = [item[1] for item in batch]
            r_batch = [item[2] for item in batch]
            next_s_batch = [item[3] for item in batch]
            y_batch = []

            next_q_batch = output.eval(feed_dict={s: next_s_batch})
            for i in range(BATCH_SIZE):
                terminal = batch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(next_q_batch[i]))

            train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_batch})

        # 更新state
        state = state_
        t += 1
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        if t % 10000 == 0:
            saver.save(sess, './saved_networks/' + GAME, global_step=t)

        # 输出信息
        state_str = ''
        if t <= OBSERVE:
            state_str = 'OBSERVE'
        else:
            state_str = 'TRAIN'

        print("TIMESTEP", t, "/ STATE", state_str, "/ EPSILON", epsilon, "/ ACTION", action_index)


def play_game():
    sess = tf.InteractiveSession()
    s, output, h_fc1 = create_network()
    train_network(s, output, h_fc1, sess)


# 主函数
def main():
    play_game()


if __name__ == '__main__':
    main()
