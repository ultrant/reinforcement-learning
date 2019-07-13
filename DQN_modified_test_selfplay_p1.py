"""
This part of code is the Deep Q Network (DQN) brain.
view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf
import cv2

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetworkP1:
    def __init__(
            self,
            mode,
            n_actions=7,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            crop_size_x=60,
            crop_size_y=60,
            n_features=60 * 60,
            output_graph=True,

    ):
        self.n_actions = n_actions
        print("-------------\n")
        print(self.n_actions)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.n_features = self.crop_size_x * self.crop_size_y
        self.mode = mode
        self.save_step = 500



        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            writer = tf.summary.FileWriter("logs/", self.sess.graph)
            writer.close()
        self.saver = tf.train.Saver(tf.global_variables())

        checkpoint = tf.train.latest_checkpoint('./checkpoints_p1')
        if checkpoint != None and self.mode == 'use mode':
            print("Restore checkpoint %s"%(checkpoint))
            self.saver.restore(self.sess, checkpoint)
        else:
            self.sess.run(tf.global_variables_initializer())
            print("Initialized New Graph")
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        #self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s = tf.placeholder(tf.float32, [None, self.crop_size_x, self.crop_size_y, 1], name='s')  # input State
        #self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.s_ = tf.placeholder(tf.float32, [None, self.crop_size_x, self.crop_size_y, 1], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        # with tf.variable_scope('eval_net'):
        #     e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='e1')
        #     self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='q')

        # with tf.variable_scope('eval_net'):
        #     conv1 = tf.layers.conv2d(inputs=self.s, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='ec1')
        #     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='ep1')
        #     conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='ec2')
        #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='ep2')
        #     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        #     dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        #     #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        #     logits = tf.layers.dense(inputs=dense, units=7)
        #     self.q_eval = tf.nn.softmax(logits, name='q')

        with tf.variable_scope('eval_net'):
            W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 1, 32], stddev=0.02))
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
            W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
            b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))
            W_fc5 = tf.Variable(tf.truncated_normal([512, self.n_actions], stddev=0.02))
            b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.n_actions]))
          # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
            conv1 = tf.nn.relu(tf.nn.conv2d(self.s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)
            conv3_flat = tf.reshape(conv3, [-1, 1024])
            fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
            fc5 = tf.matmul(fc4, W_fc5) + b_fc5
            self.q_eval = tf.nn.softmax(fc5)
            #self.q_eval = fc5


        # ------------------ build target_net ------------------
        # with tf.variable_scope('target_net'):
        #     t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
        #                          bias_initializer=b_initializer, name='t1')
        #     self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
        #                                   bias_initializer=b_initializer, name='t2')

        # with tf.variable_scope('target_net'):
        #     conv1_t = tf.layers.conv2d(inputs=self.s_, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='tc1')
        #     pool1_t = tf.layers.max_pooling2d(inputs=conv1_t, pool_size=[2, 2], strides=2, name='tp1')
        #     conv2_t = tf.layers.conv2d(inputs=pool1_t, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='tc2')
        #     pool2_t = tf.layers.max_pooling2d(inputs=conv2_t, pool_size=[2, 2], strides=2, name='tp2')
        #
        #     pool2_flat_t = tf.reshape(pool2_t, [-1, 7 * 7 * 64])
        #     dense_t = tf.layers.dense(inputs=pool2_flat_t, units=1024, activation=tf.nn.relu)
        #     # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        #     logits_t = tf.layers.dense(inputs=dense_t, units=7)
        #     self.q_next = tf.nn.softmax(logits_t, name='t')
        #
        #     #self.q_next = tf.layers.dense(inputs=pool2_t, units=self.n_actions, activation=tf.nn.relu, name='t')


        with tf.variable_scope('target_net'):
            W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 1, 32], stddev=0.02))
            b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
            b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
            W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
            b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
            W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
            b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))
            W_fc5 = tf.Variable(tf.truncated_normal([512, self.n_actions], stddev=0.02))
            b_fc5 = tf.Variable(tf.constant(0.01, shape=[self.n_actions]))
          # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
            conv1 = tf.nn.relu(tf.nn.conv2d(self.s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
            conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)
            conv3_flat = tf.reshape(conv3, [-1, 1024])
            fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
            fc5 = tf.matmul(fc4, W_fc5) + b_fc5
            self.q_next = tf.nn.softmax(fc5)
            #self.q_next = fc5



        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
       # print("all type:", s.type, a.type, r.type, s_.type)
       # print("all shape:", s.shape, a.shape, r.shape, s_.shape)
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        #print("observation1:", observation.shape)
        #observation = observation.reshape(1, -1)
        # to have batch dimension when feed into tf placeholder
        #observation = observation[np.newaxis, :]
        observation = observation[:,:, np.newaxis]
        observation = observation[np.newaxis, :, :]
        #print("observation2:", observation.shape)
        #print("self.s:", self.s.shape)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
         #   print("action_value:", actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self, step):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        #print("batch_memory:", batch_memory.shape)

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features].reshape(self.batch_size, 60, 60, 1),
                #self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                #self.s_: batch_memory[:, -self.n_features:],
                self.s_: batch_memory[:, -self.n_features:].reshape(self.batch_size, 60, 60, 1),
            })

        if self.mode == 'train' and (step % self.save_step == 0):
        #if step % self.save_step == 0:
            self.saver.save(self.sess, './checkpoints/model.ckpt', global_step=step)
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    # def pre_process(self, frame):
    #     frame = cv2.cvtColor(cv2.resize(frame, (60, 60)), cv2.COLOR_BGR2GRAY)
    #     ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    #     frame = np.reshape(frame, (60, 60, 1))
    #     return frame



if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)