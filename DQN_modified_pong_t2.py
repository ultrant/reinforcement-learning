"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf

import random  # random
import os
from collections import deque  # queue data structure. fast appends. and pops. replay memory
from numpy.random import choice
import cv2
import matplotlib.pyplot as plt

#np.random.seed(1)
#tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            # n_features,
            # learning_rate=0.01,
            # reward_decay=0.9,
            # e_greedy=0.9,
            # replace_target_iter=300,
            # memory_size=500,
            # batch_size=32,
            # e_greedy_increment=None,
            output_graph=False,
    ):
        self.ACTIONS = n_actions
        self.INITIAL_EPSILON = 1.0
        self.FINAL_EPSILON = 0.05
        # how many frames to anneal epsilon
        self.EXPLORE = 10000
        self.OBSERVE = 1000

        self.REPLAY_MEMORY = 200000
        self.BATCH = 48
        self.GAMMA = 0.99

        self.D = deque()
        self.epsilon = self.INITIAL_EPSILON
        self.argmax_t = np.zeros([self.ACTIONS])

        self.inp, self.out = self.create_graph()
        print("inp, out shape", self.inp.shape, self.out.shape)
        print("--" * 30)

     #   -------------------------------------

        # print("-------------\n")
        # print(self.n_actions)
        # self.n_features = n_features
        # self.lr = learning_rate
        # self.gamma = reward_decay
        # self.epsilon_max = e_greedy
        # self.replace_target_iter = replace_target_iter
        # self.memory_size = memory_size
        # self.batch_size = batch_size
        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        #
        # # total learning step
        # self.learn_step_counter = 0
        #
        # # initialize zero memory [s, a, r, s_]
        # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        #
        # # consist of [target_net, evaluate_net]
        # self._build_net()
        #
        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        #
        # with tf.variable_scope('hard_replacement'):
        #     self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.graph_pre()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def create_graph(self):
        # input for pixel data
        s = tf.placeholder("float", [None, 60, 60, 4])

        W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 4, 32], stddev=0.02))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
        W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
        b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))
        W_fc5 = tf.Variable(tf.truncated_normal([512, self.ACTIONS], stddev=0.02))
        b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS]))

        # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
        conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)
        conv3_flat = tf.reshape(conv3, [-1, 1024])
        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
        fc5 = tf.matmul(fc4, W_fc5) + b_fc5
        return s, fc5

        ## --------------------------------------------------------------
        #
        # # ------------------ all inputs ------------------------
        # self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        # self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        # self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        #
        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        #
        # # ------------------ build evaluate_net ------------------
        # with tf.variable_scope('eval_net'):
        #     e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='e1')
        #     self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='q')
        #
        # # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #     t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='t1')
        #     self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='t2')
        #
        # with tf.variable_scope('q_target'):
        #     q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
        #     self.q_target = tf.stop_gradient(q_target)
        # with tf.variable_scope('q_eval'):
        #     a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
        #     self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        # with tf.variable_scope('loss'):
        #     self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        # with tf.variable_scope('train'):
        #     self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def graph_pre(self):
        # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
        self.argmax = tf.placeholder("float", [None, self.ACTIONS])
        self.gt = tf.placeholder("float", [None])  # ground truth
        #  global_step = tf.Variable(0, name='global_step')

        # action
        self.action = tf.reduce_sum(tf.multiply(self.out, self.argmax), reduction_indices=1)
        # cost function we will reduce through backpropagation
        self.cost = tf.reduce_mean(tf.square(self.action - self.gt))
        # optimization fucntion to reduce our minimize our cost function
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def store_transition(self, inp_t, a, reward_t, inp_t1):
        self.D.append((inp_t, self.argmax_t, reward_t, inp_t1))
        if len(self.D) > self.REPLAY_MEMORY:
            self.D.popleft()
        #
        # ## --------------------------
        # if not hasattr(self, 'memory_counter'):
        #     self.memory_counter = 0
        # transition = np.hstack((s, [a, r], s_))
        # # replace the old memory with new memory
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        # self.memory_counter += 1

    def choose_action(self, observation):
        #inp, out = self.create_graph()
        out_t = self.sess.run(
             [self.out],
             feed_dict={self.inp: [observation]})
        out_t = out_t[0]

        #out_t = out.eval(feed_dict={inp: [inp_t]})[0]
        # argmax function
        self.argmax_t = np.zeros([self.ACTIONS])

        #
        if (random.random() <= self.epsilon):
            # make 0 the most choosen action for realistic randomness
            #maxIndex = choice((0, 1, 2), 1, p=(0.90, 0.05, 0.05))
            maxIndex = choice(range(self.ACTIONS), 1)
            maxIndex = maxIndex[0]
           # print("random action:", maxIndex)
        else:
            maxIndex = np.argmax(out_t)
           # print("max action:", maxIndex)
        self.argmax_t[maxIndex] = 1

        if self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE

        return maxIndex
        #
        # ###  ---------------------------------------
        # # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        #
        # if np.random.uniform() < self.epsilon:
        #     # forward feed the observation and get q value for every actions
        #     actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        #     action = np.argmax(actions_value)
        # else:
        #     action = np.random.randint(0, self.n_actions)
        # return action

    def learn(self,step):

      #   # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
      #   argmax = tf.placeholder("float", [None, self.ACTIONS])
      #   gt = tf.placeholder("float", [None])  # ground truth
      # #  global_step = tf.Variable(0, name='global_step')
      #
      #   # action
      #   action = tf.reduce_sum(tf.multiply(self.out, argmax), reduction_indices=1)
      #   # cost function we will reduce through backpropagation
      #   cost = tf.reduce_mean(tf.square(action - gt))
      #   # optimization fucntion to reduce our minimize our cost function
      #   train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
      #
      #   init = tf.global_variables_initializer()
      #   self.sess.run(init)

        minibatch = random.sample(self.D, self.BATCH)
       # print("minibatch shape:", len(minibatch), len(minibatch[0]), len(minibatch[1]), len(minibatch[2]),len(minibatch[3]))

        inp_batch = [d[0] for d in minibatch]
        argmax_batch = [d[1] for d in minibatch]
        reward_batch = [d[2] for d in minibatch]
        inp_t1_batch = [d[3] for d in minibatch]

        gt_batch = []
        out_batch = self.sess.run(
             [self.out],
             feed_dict={self.inp: inp_t1_batch})
       # print("out_batch shape:", out_batch[0])
       # print("reward_batch shape:", len(reward_batch))

       # out_batch = out.eval(feed_dict={inp: inp_t1_batch})
        for i in range(0, len(minibatch)):
            gt_batch.append(reward_batch[i] + self.GAMMA * np.max(out_batch[0][i]))

        #bb_out, bb_argmax, bb_action, bb_cost, bb_train_step = self.sess.run([self.out, argmax, action, cost, train_step],
        bb_cost, bb_train_step = self.sess.run([self.cost, self.train_step],
                                                                        feed_dict={
                                                                            self.gt: gt_batch,
                                                                            self.argmax: argmax_batch,
                                                                            self.inp: inp_batch
                                                                        })
     #   print("tf bb_out, bb_argmax:", bb_out, bb_argmax)
      #  print("tf bb_action, bb_cost, bb_train_step:", bb_action, bb_cost, bb_train_step)

       #### -------------------------------------
       #
       #  # check to replace target parameters
       #  if self.learn_step_counter % self.replace_target_iter == 0:
       #      self.sess.run(self.target_replace_op)
       #      print('\ntarget_params_replaced\n')
       #
       #  # sample batch memory from all memory
       #  if self.memory_counter > self.memory_size:
       #      sample_index = np.random.choice(self.memory_size, size=self.batch_size)
       #  else:
       #      sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
       #  batch_memory = self.memory[sample_index, :]
       #  ss = batch_memory[:, :self.n_features]
       #  aa = batch_memory[:, self.n_features]
       #  rr = batch_memory[:, self.n_features + 1]
       #  ss_ = batch_memory[:, -self.n_features:]
       #  print("ss,aa,rr,ss_:", ss.shape, aa, rr, ss_.shape)
       #
       # # _, cost = self.sess.run(
       # #     [self._train_op, self.loss],
       # #     feed_dict={
       #  self_q_target, self_q_next, self_q_eval_wrt_a, a_indice, self_q_eval, self_a, cost = self.sess.run(
       #      [self.q_target, self.q_next, self.q_eval_wrt_a,
       #       tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1), self.q_eval, self.a, self.loss],
       #      feed_dict={
       #          self.s: batch_memory[:, :self.n_features],
       #          self.a: batch_memory[:, self.n_features],
       #          self.r: batch_memory[:, self.n_features + 1],
       #          self.s_: batch_memory[:, -self.n_features:],
       #      })
       #
       #  print("self_q_target, self_q_eval_wrt_a :", self_q_target, self_q_eval_wrt_a)
       #  print("self_q_next", self_q_next)
       #  print("a_indice:", a_indice)
       #  print("self_q_eval, self_a, cost :", self_q_eval, self_a, cost)
       #
       #  self.cost_his.append(cost)
       #
       #  # increasing epsilon
       #  self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
       #  self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def pre_process(self, frame, crop_size):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
        return frame

    def plt_data(self,image_path, win_rate_list_p1, win_rate_list_p2, avg_step_p1_list, avg_step_p2_list, reward_p1_list, reward_p2_list):
        plt.figure(1)
        plt.plot(np.arange(len(win_rate_list_p1)), win_rate_list_p1, label='player1_win_rate(random)')
        plt.plot(np.arange(len(win_rate_list_p2)), win_rate_list_p2, label='player2_win_rate(AI)')
        plt.ylabel('Winning rate')
        plt.xlabel('game numbers')
        plt.legend(loc='upper left')
        fig_name1=image_path + 'win_rate.png'
        #if os.path.exists(fig_name1):
        #    os.rename(fig_name1,fig_name1 + "1")
        plt.savefig(fig_name1)
        plt.show()
       # figure;
        plt.figure(2)
        plt.plot(np.arange(len(avg_step_p1_list)), avg_step_p1_list, label='player1_aver_step(random)')
        plt.plot(np.arange(len(avg_step_p2_list)), avg_step_p2_list, label='player2_aver_step(AI)')
        plt.ylabel('Average steps ')
        plt.xlabel('game numbers')
        plt.legend(loc='upper left')
        plt.savefig(image_path + 'aver_step.png')
        plt.show()
        plt.figure(3)
        plt.plot(np.arange(len(reward_p1_list)), reward_p1_list, label='player1_reward_total(random)')
        plt.plot(np.arange(len(reward_p2_list)), reward_p2_list, label='player2_reward_total(AI)')
        plt.ylabel('Total reward ')
        plt.xlabel('game numbers')
        plt.legend(loc='upper left')
        plt.savefig(image_path + 'reward_total.png')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)