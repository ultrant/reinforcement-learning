from pong_dqn import PongGame
from DQN_modified_pong_t2 import DeepQNetwork
import cv2
import numpy as np
import os
import csv


def run_connect4():
    step = 0
    reward_total = 0
    info_total = [0, 0]
    crop_size = (60, 60)
    game_number = 0
    reward_list = []
    win_rate_list = []
    project_path = 'D:\zhaomi\code\project\deep_q_network'
    csv_file = open(project_path + '\csv_data\\result_data1.csv', 'a', newline='')
    writer = csv.writer(csv_file)
   # inp, out = RL.create_graph()


    #-----------------------
    frame = env.getPresentFrame()
    frame = RL.pre_process(frame, crop_size)
    # print("frame shape::", frame.shape)
    observation = np.stack((frame, frame, frame, frame), axis=2)

    for episode in range(40000):
   # if(game_number <= 20000):
    # every game number
    #while True:
        game_number += 1
        print("--------------------- game numbers:", game_number, " ---------------------\n")
        # initial observation

#--------------------------------------------

       # # # cv2.imwrite(image_path + "%d_p1.png" % i, observation_p1)

        #every step number
        while True:
            # fresh env
            # RL choose action based on observation
          #  print("observation shape:", observation.shape)
         #   print("--------------------------")
            #print("observation shape:", observation.shape[2])
            #for k in range(observation.shape[2]):
            #cv2.imshow('a.png', observation[:, :, 0])
            #cv2.waitKey(0)
            #print("step:", step)
            action = RL.choose_action(observation)
          #  print("type of action", type(action), action)
           # action = action[0]
           # print("action:", action)
            info, done, reward, observation_n = env.getNextFrame(action, [step])
           # print("info, done, reward, observation_n :", info, done, reward)
            step += 1
            reward_total += reward
            info_total = np.sum([info_total, info], axis=0)

            observation_n = RL.pre_process(observation_n, crop_size)
            observation_n = np.reshape(observation_n, (60, 60, 1))
            observation_n = np.append(observation_n, observation[:, :, 0:3], axis=2)

            RL.store_transition(observation, action, reward, observation_n)

            # observation_flatten = observation.flatten()
            # observation_n_flatten = observation_n.flatten()
            # RL.store_transition(observation_flatten, action, reward, observation_n_flatten)

            if (step > 200) and (step % 1 == 0):
                RL.learn(step)

            observation = observation_n

            if done:
                break

        print("\n", "*" * 80)
        print("game_number: %d end: \nreward_total: %d, info_p1_total[n_w,n_l]: %s, step: %d " % (game_number, reward_total, str(info_total), step))
        print("*" * 80, "\n")

        win_rate = round(info_total[0] / (sum(info_total)), 2)
        win_rate_list.append(win_rate)
        reward_list.append(reward_total)

        csv_data = [win_rate, reward_total]
        if game_number % 100 == 0:
             writer.writerow(csv_data)
        if game_number % 1000 == 0:
             RL.plt_data(project_path + "\images\\", win_rate_list, [], [], [], reward_list, [])

#
#
# #-------------------------------------------------
#             observation_p1 = RL.pre_process(observation_p1, crop_size)
#         #    cv2.imwrite(image_path + "%d_p1_pre.png" % i, observation_p1)
#             action_p2 = RL.choose_action(observation_p1)
#             # RL take action and get next observation and reward
#             draw_info_list = [game_number, reward_p2_total, step_p2]
#             observation_p2, reward_p2, done_p2, info_p2 = env.makeMove_P2(player, action_p2, draw_info_list, is_draw)
#             step_p2 += 1
#             reward_p2_total += reward_p2[1]
#             reward_p1_total += reward_p2[0]
#             info_p2_total = np.sum([info_p2_total, info_p2[-4:]], axis=0)
#             info_p1_total = np.sum([info_p1_total, info_p2[:4]], axis=0)
#          #    print("【action_p2: %d,  done_p2: %s】:\nreward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(info_p2[4], str(done_p2), reward_p2[1], reward_p2_total , str(info_p2[-4:])))
#          #    print("reward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" %(reward_p2[0], reward_p1_total, str(info_p2[:4])))
#          #
#          # #   print("action_p2: %d, reward_p2: %d, reward_p2_total: %d, done_p2: %s, info_p2[n_w, n_d, n_l, n_il]: %s" %(info_p2[4], reward_p2, reward_p2_total, str(done_p2), str(info_p2[:4])))
#          # #   print("chessboard:\n", info_p2[5])
#          #  #  cv2.imwrite(image_path + "%d_p2.png" % i, observation_p2)
#             i = i + 1
#             if done_p2:
#                 break
#
#             observation_p2 = RL.pre_process(observation_p2, crop_size)
#          #   cv2.imwrite(image_path + "%d_p2_pre.png" % i, observation_p2)
#             observation_p1 = observation_p1.flatten()
#             observation_p2 = observation_p2.flatten()
#             RL.store_transition(observation_p1, action_p2, reward_p2[1], observation_p2)
#
#             if (step_p2 > 200) and (step_p2 % 1 == 0):
#                 RL.learn(step_p2)
#
#             observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
#             step_p1 += 1
#             reward_p1_total += reward_p1[0]
#             reward_p2_total += reward_p1[1]
#             info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
#             info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)
#           #   print("【action_p1: %d,  done_p1: %s】:\nreward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" % (info_p1[4], str(done_p1), reward_p1[0], reward_p1_total , str(info_p1[:4])))
#           #   print("reward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(reward_p1[1], reward_p2_total, str(info_p1[-4:])))
#           # #  print("action_p1: %d, reward_p1: %d, reward_p1_total: %d, done_p1: %s, info_p1[n_w, n_d, n_l, n_il]: %s" %(info_p1[4], reward_p1, reward_p1_total, str(done_p1), str(info_p1[:4])))
#           # #  print("chessboard:\n", info_p1[5])
#             # swap observation
#
#             # break while loop when end of this episode
#            # cv2.imwrite(image_path + "%d_p1.png" % i, observation_p1)
#             i = i + 1
#             if done_p1:
#                 break
#
#
#        # print("game_number: %d, reward_p2_total: %d, info_p2_total[n_w, n_d, n_l, n_il]: %s, reward_p1_total: %d, info_p1_total[n_w, n_d, n_l, n_il]: %s, step_p2: %d, step_p1:  %d .\n" % (game_number, reward_p2_total, str(info_p2_total), reward_p1_total, str(info_p1_total), step_p2, step_p1))
#         win_rate_p1 = round(info_p1_total[0] / (sum(info_p1_total[:4])), 2)
#         win_rate_p2 = round(info_p2_total[0] / (sum(info_p2_total[:4])), 2)
#         avg_step_p1 = round(step_p1/ game_number, 2)
#         avg_step_p2 = round(step_p2/ game_number, 2)
#        # print("avg_step_p1, p2:", avg_step_p1, avg_step_p2)
#        # print(win_rate_p1, win_rate_p2)
#         win_rate_p1_list.append(win_rate_p1)
#         win_rate_p2_list.append(win_rate_p2)
#         avg_step_p1_list.append(avg_step_p1)
#         avg_step_p2_list.append(avg_step_p2)
#         reward_p1_list.append(reward_p1_total)
#         reward_p2_list.append(reward_p2_total)
#
#         print("\n", "*" * 80)
#         print("game_number: %d end: \nreward_p1_total: %d, info_p1_total[n_w, n_d, n_l, n_il]: %s, step_p1: %d " % (game_number, reward_p1_total, str(info_p1_total), step_p1))
#         print("reward_p2_total: %d, info_p2_total[n_w, n_d, n_l, n_il]: %s, step_p2: %d." % (reward_p2_total, str(info_p2_total), step_p2))
#         print("*" * 80, "\n")
#
#         csv_data = [win_rate_p1, win_rate_p2, avg_step_p1, avg_step_p2, reward_p1_total, reward_p2_total]
#         if game_number % 3 == 0:
#             writer.writerow(csv_data)
#         if game_number % 3 == 0:
#             RL.plt_data(project_path + "\images\\", win_rate_p1_list, win_rate_p2_list, avg_step_p1_list, avg_step_p2_list, reward_p1_list, reward_p2_list)
#

    # end of game
    #print('win_rate_p1_list, win_rate_p2_list',win_rate_p1_list, win_rate_p2_list)
    #print('avg_step_p1_list, avg_step_p2_list', avg_step_p1_list, avg_step_p2_list)
    #print('reward_p1_list,reward_p2_list', reward_p1_list, reward_p2_list)
    print('game over')
    # csv_file.close()


if __name__ == "__main__":
    # maze game
    env = PongGame()
    mode_type = 'train'
    #mode_type = 'use mode'
    # RL = DeepQNetwork(mode=mode_type, n_actions=3,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,
    #                   memory_size=2000,
    #                   output_graph=True
    #                   )
    RL = DeepQNetwork(n_actions=3,
                      output_graph=False
                      )
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./csv_data'):
        os.makedirs('./csv_data')
    if not os.path.exists('./images'):
        os.makedirs('./images')
    run_connect4()

    RL.plot_cost()
