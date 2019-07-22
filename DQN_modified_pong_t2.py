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
from tensorflow.python.tools import inspect_checkpoint as chkp


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            mode,
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
        self.mode = mode
        self.INITIAL_EPSILON = 1.0
        self.FINAL_EPSILON = 0.05
        # how many frames to anneal epsilon
        self.EXPLORE = 10000
        self.OBSERVE = 1000

        self.REPLAY_MEMORY = 200000
        self.BATCH = 48
        self.GAMMA = 0.99
        self.SAVE_STEP = 5000
       # self.cost_his = []

        self.D = deque()
        self.epsilon = self.INITIAL_EPSILON
        self.argmax_t = np.zeros([self.ACTIONS])

        self.inp, self.out = self.create_graph()

        self.graph_pre()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

       # self.sess.run(tf.global_variables_initializer())
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

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())

        checkpoint = tf.train.latest_checkpoint('./checkpoints')
        if checkpoint != None and self.mode == 'use mode':
            print('Restore Checkpoint %s' % (checkpoint))
            self.saver.restore(self.sess, checkpoint)
           # self.saver.restore(self.sess, './checkpoints/model.ckpt-713000')
          #  chkp.print_tensors_in_checkpoint_file(checkpoint, tensor_name='Variable_9', all_tensors=False)
            print("Model restored.")
        else:
           # b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.ACTIONS]))
            init = tf.global_variables_initializer()
            self.sess.run(init)
            #print(self.sess.run(b_fc5))
           # print(self.sess.run(self.out))
            print("Initialized new Graph")

    def store_transition(self, inp_t, a, reward_t, inp_t1):
        self.D.append((inp_t, self.argmax_t, reward_t, inp_t1))
        if len(self.D) > self.REPLAY_MEMORY:
            self.D.popleft()


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


    def learn(self,step):


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
        if step % self.SAVE_STEP == 0:
            self.saver.save(self.sess, './checkpoints/model.ckpt', global_step=step)
        #if step % 100 == 0:
        #    self.cost_his.append(bb_cost)
        return bb_cost

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

    def plt_data(self,image_path, win_rate_list_p1, win_rate_list_p2, avg_step_p1_list, avg_step_p2_list, reward_p1_list, reward_p2_list, loss_p1_list, loss_p2_list):
        plt.figure(1)
        plt.plot(np.arange(len(win_rate_list_p1)), win_rate_list_p1, label='player1_win_rate(random)')
        plt.plot(np.arange(len(win_rate_list_p2)), win_rate_list_p2, label='player2_win_rate(AI)')
        plt.ylabel('Winning rate')
        plt.xlabel('game number')
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
        plt.xlabel('game number')
        plt.legend(loc='upper left')
        plt.savefig(image_path + 'aver_step.png')
        plt.show()
        plt.figure(3)
        plt.plot(np.arange(len(reward_p1_list)), reward_p1_list, label='player1_reward_total(random)')
        plt.plot(np.arange(len(reward_p2_list)), reward_p2_list, label='player2_reward_total(AI)')
        plt.ylabel('Total reward ')
        plt.xlabel('game number')
        plt.legend(loc='upper left')
        plt.savefig(image_path + 'reward_total.png')
        plt.show()
        plt.figure(4)
        plt.plot(np.arange(len(loss_p1_list)), loss_p1_list, label='player1_loss(random)')
        plt.plot(np.arange(len(loss_p2_list)), loss_p2_list, label='player2_loss(AI)')
        plt.ylabel('Loss ')
        plt.xlabel('game number')
        plt.legend(loc='upper left')
        plt.savefig(image_path + 'loss.png')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)