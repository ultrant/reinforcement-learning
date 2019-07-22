from connect4_selfplay import Connect4
from DQN_modified_selfplay_p1_v2 import DeepQNetworkP1
from DQN_modified_selfplay_p2_v2 import DeepQNetworkP2
#from DQN_modified_test import DeepQNetwork
import cv2
import numpy as np
import os
import csv
import datetime
import time

# def prepare_dirs(projectDir):
#     if not os.path.exists(projectDir+"\\csv_data"):
#         os.makedirs(projectDir+"\\csv_data")
#     if not os.path.exists(projectDir+"\\images"):
#         os.makedirs(projectDir+"\\images")

def run_connect4():
    step_p1 = 0
    step_p2 = 0
    player_p1 = 1
    player_p2 = 2
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

    loss_p1_list = []
    loss_p2_list = []
    loss_p1 = 0
    loss_p2 = 0

    draw_info_list = []

    project_path = 'D:\zhaomi\code\project\deep_q_network'
    csv_file = open(project_path + '\csv_data\\result_data_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv', 'a', newline='')

    writer = csv.writer(csv_file)

    i = 0
    #for episode in range(40000):
    for episode in range(150000):
    # every game number
    #while True:
        game_number += 1
        print("--------------------- game numbers:", game_number, " ---------------------\n")
        # initial observation
        observation_0 = env.reset()
        observation_0 = RL1.pre_process(observation_0, crop_size)
        observation_0 = np.stack((observation_0, observation_0, observation_0, observation_0), axis=2)
       # print("obser_0 shape:",observation_0.shape)
        action_p1 = RL1.choose_action(observation_0)
        observation_p1, reward_p1, done_p1, info_p1 = env.makeMove_P1(player_p1, action_p1, draw_info_list, is_draw)
        step_p1 += 1
        reward_p1_total += reward_p1[0]
        reward_p2_total += reward_p1[1]
        info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
        info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)
        observation_p1 = RL1.pre_process(observation_p1, crop_size)
        observation_p1 = np.stack((observation_p1, observation_p1, observation_p1, observation_p1), axis=2)





    # ##--------------------------------
    #     observation_p1, reward_p1, done_p1, info_p1 = env.makeMove_P1(player_p1, action_p1, draw_info_list, is_draw)
    #     observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
    #     step_p1 += 1
    #     reward_p1_total += reward_p1[0]
    #     reward_p2_total += reward_p1[1]
    #     info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
    #     info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)
    #     observation_p1 = RL.pre_process(observation_p1, crop_size)
    #    # observation_p1 = np.reshape(observation_p1, (60, 60, 1))
    #     observation_p1 = np.stack((observation_p1, observation_p1, observation_p1, observation_p1), axis=2)
    #    #  print("【action_p1: %d,  done_p1: %s】:\nreward_p1: %d, reward_p1_total: %d, info_p1[n_w, n_d, n_l, n_il]: %s" %(info_p1[4], str(done_p1), reward_p1[0], reward_p1_total , str(info_p1[:4])))
    #    #  print("reward_p2: %d, reward_p2_total: %d, info_p2[n_w, n_d, n_l, n_il]: %s" %(reward_p1[1], reward_p2_total, str(info_p1[-4:])))
    #    # # print("chessboard:\n", info_p1[5])
    #    # # cv2.imwrite(image_path + "%d_p1.png" % i, observation_p1)
    #     i = i + 1

        #every step number
        while True:
            # fresh env
            # RL choose action based on observation

        #    cv2.imwrite(image_path + "%d_p1_pre.png" % i, observation_p1)
            action_p2 = RL2.choose_action(observation_p1)
            # RL take action and get next observation and reward
            draw_info_list = [game_number, reward_p2_total, step_p2]
            observation_p2, reward_p2, done_p2, info_p2 = env.makeMove_P2(player_p2, action_p2, draw_info_list, is_draw)
            step_p2 += 1
            reward_p2_total += reward_p2[1]
            reward_p1_total += reward_p2[0]
            info_p2_total = np.sum([info_p2_total, info_p2[-4:]], axis=0)
            info_p1_total = np.sum([info_p1_total, info_p2[:4]], axis=0)

            observation_p2 = RL2.pre_process(observation_p2, crop_size)
            observation_p2 = np.reshape(observation_p2, (60, 60, 1))
            observation_p2 = np.append(observation_p2, observation_p1[:, :, 0:3], axis=2)
            RL2.store_transition(observation_p1, action_p2, reward_p2[1], observation_p2)

           # i = i + 1
            if done_p2:
                break
            if (step_p2 > 200) and (step_p2 % 1 == 0):
                loss_p2 = RL2.learn(step_p2)

            action_p1 = RL1.choose_action(observation_p2)
            observation_p1, reward_p1, done_p1, info_p1 = env.makeMove_P1(player_p1, action_p1, draw_info_list, is_draw)
            ###get reward_p1 from env ,it includes [reward_player1, reward_player2]
            #observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
            step_p1 += 1
            reward_p1_total += reward_p1[0]
            reward_p2_total += reward_p1[1]
            info_p1_total = np.sum([info_p1_total, info_p1[:4]], axis=0)
            info_p2_total = np.sum([info_p2_total, info_p1[-4:]], axis=0)

            observation_p1 = RL1.pre_process(observation_p1, crop_size)
            observation_p1 = np.reshape(observation_p1, (60, 60, 1))
            observation_p1 = np.append(observation_p1, observation_p2[:, :, 0:3], axis=2)
            RL1.store_transition(observation_p2, action_p1, reward_p1[0], observation_p1)

            i = i + 1
            if done_p1:
                break
            if (step_p1 > 200) and (step_p1 % 1 == 0):
                loss_p1 = RL1.learn(step_p1)


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
        loss_p2_list.append(loss_p2)
        loss_p1_list.append(loss_p1)

        print("\n", "*" * 80)
        print("game_number: %d end: \nreward_p1_total: %d, info_p1_total[n_w, n_d, n_l, n_il]: %s, step_p1: %d, loss_p1:%s " % (game_number, reward_p1_total, str(info_p1_total), step_p1, str(loss_p1)))
        print("reward_p2_total: %d, info_p2_total[n_w, n_d, n_l, n_il]: %s, step_p2: %d, loss_p2: %s." % (reward_p2_total, str(info_p2_total), step_p2, str(loss_p2)))
        print("*" * 80, "\n")

        csv_data = [win_rate_p1, win_rate_p2, avg_step_p1, avg_step_p2, reward_p1_total, reward_p2_total, info_p1_total[0], info_p1_total[1], info_p1_total[2], info_p1_total[3], info_p2_total[0], info_p2_total[1], info_p2_total[2], info_p2_total[3], loss_p1, loss_p2]
        if game_number % 1000 == 0:
            writer.writerow(csv_data)
            csv_file.flush()

        if game_number % 10000 == 0:
            RL1.plt_data(project_path + "\images\\", win_rate_p1_list, win_rate_p2_list, avg_step_p1_list, avg_step_p2_list, reward_p1_list, reward_p2_list,loss_p1_list, loss_p2_list)
         #   RL.plot_cost()


    # end of game
    #print('win_rate_p1_list, win_rate_p2_list',win_rate_p1_list, win_rate_p2_list)
    #print('avg_step_p1_list, avg_step_p2_list', avg_step_p1_list, avg_step_p2_list)
    #print('reward_p1_list,reward_p2_list', reward_p1_list, reward_p2_list)
    print('game over')
    csv_file.close()


def test_pygame():
    for _ in range(10000):
        env.draw_board()
        env.mute_board()
        time.sleep(0.1)

# def pre_process(frame, crop_size):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
#     return frame

if __name__ == "__main__":
    # maze game
    start_time = datetime.datetime.now()
    print("start time:" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
    env = Connect4()
    mode_type = 'train'
    #mode_type = 'use mode'
    RL1 = DeepQNetworkP1(n_actions=7,
                      mode=mode_type,
                      output_graph=False
                      )
    RL2 = DeepQNetworkP2(n_actions=7,
                      mode=mode_type,
                      output_graph=False
                      )
    if not os.path.exists('./checkpoints_p1'):
        os.makedirs('./checkpoints_p1')
    if not os.path.exists('./checkpoints_p2'):
        os.makedirs('./checkpoints_p2')
    if not os.path.exists('./csv_data'):
        os.makedirs('./csv_data')
    if not os.path.exists('./images'):
        os.makedirs('./images')
    run_connect4()

  #  RL.plot_cost()
    end_time = datetime.datetime.now()
    print("end time:" + end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("run time(seconds):" + str((end_time - start_time).seconds))
