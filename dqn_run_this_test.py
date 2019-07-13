#from connect4game import Connect4
from connect4_selfplay import Connect4
from DQN_modified_test import DeepQNetwork
import cv2
import numpy as np
import os


def run_connect4():
    step_p1 = 0
    step_p2 = 0
    player = 2
    is_draw = 1
    crop_size = (60, 60)
    reward_p1_total = 0
    reward_p2_total = 0
    game_number = 0
    info_p1_total = [0, 0, 0, 0]
    info_p2_total = [0, 0, 0, 0]
    win_rate_p1_list = []
    win_rate_p2_list = []
    avg_step_p1_list = []
    avg_step_p2_list = []
    reward_p1_list = []
    reward_p2_list = []

    image_path = 'D:\zhaomi\code\project\deep_q_network\images\\'
    i = 0
    for episode in range(50000):
    # every game number
    #while True:
        game_number += 1
        print("--------------------- game numbers:", game_number, " ---------------------\n")
        # initial observation
        env.reset()
        observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
        step_p1 += 1
        reward_p1_total += reward_p1[0]
        reward_p2_total += reward_p1[1]
        info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
        info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)
       #  print("【action_p1: %d,  done_p1: %s】:\nreward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" %(info_p1[4], str(done_p1), reward_p1[0], reward_p1_total , str(info_p1[:4])))
       #  print("reward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(reward_p1[1], reward_p2_total, str(info_p1[-4:])))
       # # print("chessboard:\n", info_p1[5])
       # # cv2.imwrite(image_path + "%d_p1.png" % i, observation_p1)
        i = i + 1

        #every step number
        while True:
            # fresh env
            # RL choose action based on observation
            observation_p1 = RL.pre_process(observation_p1, crop_size)
        #    cv2.imwrite(image_path + "%d_p1_pre.png" % i, observation_p1)
            action_p2 = RL.choose_action(observation_p1)
            # RL take action and get next observation and reward
            draw_info_list = [game_number, reward_p2_total, step_p2]
            observation_p2, reward_p2, done_p2, info_p2 = env.makeMove_P2(player, action_p2, draw_info_list, is_draw)
            step_p2 += 1
            reward_p2_total += reward_p2[1]
            reward_p1_total += reward_p2[0]
            info_p2_total = np.sum([info_p2_total, info_p2[-4:]], axis=0)
            info_p1_total = np.sum([info_p1_total, info_p2[:4]], axis=0)
         #    print("【action_p2: %d,  done_p2: %s】:\nreward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(info_p2[4], str(done_p2), reward_p2[1], reward_p2_total , str(info_p2[-4:])))
         #    print("reward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" %(reward_p2[0], reward_p1_total, str(info_p2[:4])))
         #
         # #   print("action_p2: %d, reward_p2: %d, reward_p2_total: %d, done_p2: %s, info_p2[n_w, n_d, n_l, n_il]: %s" %(info_p2[4], reward_p2, reward_p2_total, str(done_p2), str(info_p2[:4])))
         # #   print("chessboard:\n", info_p2[5])
         #  #  cv2.imwrite(image_path + "%d_p2.png" % i, observation_p2)
            i = i + 1
            if done_p2:
                break

            observation_p2 = RL.pre_process(observation_p2, crop_size)
         #   cv2.imwrite(image_path + "%d_p2_pre.png" % i, observation_p2)
            observation_p1 = observation_p1.flatten()
            observation_p2 = observation_p2.flatten()
            RL.store_transition(observation_p1, action_p2, reward_p2[1], observation_p2)

            if (step_p2 > 200) and (step_p2 % 1 == 0):
                RL.learn(step_p2)

            observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
            step_p1 += 1
            reward_p1_total += reward_p1[0]
            reward_p2_total += reward_p1[1]
            info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
            info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)
          #   print("【action_p1: %d,  done_p1: %s】:\nreward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" % (info_p1[4], str(done_p1), reward_p1[0], reward_p1_total , str(info_p1[:4])))
          #   print("reward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(reward_p1[1], reward_p2_total, str(info_p1[-4:])))
          # #  print("action_p1: %d, reward_p1: %d, reward_p1_total: %d, done_p1: %s, info_p1[n_w, n_d, n_l, n_il]: %s" %(info_p1[4], reward_p1, reward_p1_total, str(done_p1), str(info_p1[:4])))
          # #  print("chessboard:\n", info_p1[5])
            # swap observation

            # break while loop when end of this episode
           # cv2.imwrite(image_path + "%d_p1.png" % i, observation_p1)
            i = i + 1
            if done_p1:
                break
       # print("game_number: %d, reward_p2_total: %d, info_p2_total[n_w, n_d, n_l, n_il]: %s, reward_p1_total: %d, info_p1_total[n_w, n_d, n_l, n_il]: %s, step_p2: %d, step_p1:  %d .\n" % (game_number, reward_p2_total, str(info_p2_total), reward_p1_total, str(info_p1_total), step_p2, step_p1))
        win_rate_p1 = round(info_p1_total[0] / (sum(info_p1_total[:4])), 2)
        win_rate_p2 = round(info_p2_total[0] / (sum(info_p2_total[:4])), 2)
        avg_step_p1 = round(step_p1/ game_number, 2)
        avg_step_p2 = round(step_p2/ game_number, 2)
       # print("avg_step_p1, p2:", avg_step_p1, avg_step_p2)
       # print(win_rate_p1, win_rate_p2)
        win_rate_p1_list.append(win_rate_p1)
        win_rate_p2_list.append(win_rate_p2)
        avg_step_p1_list.append(avg_step_p1)
        avg_step_p2_list.append(avg_step_p2)
        reward_p1_list.append(reward_p1_total)
        reward_p2_list.append(reward_p2_total)

        print("\n", "*" * 80)
        print("game_number: %d: \nreward_p1_total: %d, info_p1_total[n_w, n_d, n_l, n_il]: %s, step_p1: %d " % (game_number, reward_p1_total, str(info_p1_total), step_p1))
        print("reward_p2_total: %d, info_p2_total[n_w, n_d, n_l, n_il]: %s, step_p2: %d." % (reward_p2_total, str(info_p2_total), step_p2))
        print("*" * 80, "\n")
           # env.draw_info(game_number, reward_p2_total, step)

    # end of game
    #print('win_rate_p1_list, win_rate_p2_list',win_rate_p1_list, win_rate_p2_list)
    #print('avg_step_p1_list, avg_step_p2_list', avg_step_p1_list, avg_step_p2_list)
    #print('reward_p1_list,reward_p2_list', reward_p1_list, reward_p2_list)
    RL.plt_win_rate(win_rate_p1_list, win_rate_p2_list, avg_step_p1_list, avg_step_p2_list, reward_p1_list, reward_p2_list)
    print('game over')

# def pre_process(frame, crop_size):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
#     return frame

if __name__ == "__main__":
    # maze game
    env = Connect4()
    mode_type = 'train'
    #mode_type = 'use mode'
    RL = DeepQNetwork(mode=mode_type, n_actions=7, n_features=60*60*1,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    run_connect4()

    RL.plot_cost()
