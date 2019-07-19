#from maze_env import Maze
from connect4_selfplay import Connect4
from DQN_modified_test_selfplay_p1 import DeepQNetworkP1
from DQN_modified_test_selfplay_p2 import DeepQNetworkP2
import cv2
import numpy as np
import os

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
    for episode in range(300):
        game_number += 1
        observation_p1 = env.reset()
        observation_p1 = pre_process(observation_p1, crop_size)
        while True:
    # every game number
   # while True:
       # print("-------------step:", step)
        # initial observation

        ### p1
       # observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)

            action_p1 = RL1.choose_action(observation_p1)
    ##### draw_info_list_p1 初始化
            draw_info_list_p1 = [game_number, reward_p1_total, step_p1]
            observation_p2, reward_p1, done_p1, info_p1 = env.makeMove_P1(player_p1, action_p1, draw_info_list_p1, is_draw)
            print(type(reward_p1_total), type(reward_p1))
            reward_p1_total += reward_p1[0]
            if done_p1:
                break
            observation_p2 = pre_process(observation_p2, crop_size)
            observation_p1_flatten = observation_p1.flatten()
            observation_p2_flatten = observation_p2.flatten()
            print("observation_p1, action_p1, reward_p1, observation_p2:",observation_p1.shape, action_p1, reward_p1, observation_p2.shape)
            RL1.store_transition(observation_p1_flatten, action_p1, reward_p1, observation_p2_flatten)
            if (step_p1 > 200) and (step_p1 % 5 == 0):
                RL1.learn(step_p1)
            step_p1 += 1


            #### p2
            #observation_p2 = pre_process(observation_p2, crop_size)
            print("observation_p2:", observation_p2.shape)
            action_p2 = RL2.choose_action(observation_p2)
            draw_info_list_p2 = [game_number, reward_p2_total, step_p2]
            observation_p1, reward_p2, done_p2, info_p2 = env.makeMove_P2(player_p2, action_p2, draw_info_list_p2, is_draw)
            reward_p2_total += reward_p2
            if done_p2:
                break
            observation_p1 = pre_process(observation_p1, crop_size)
            observation_p1_flatten = observation_p1.flatten()
            observation_p2_flatten = observation_p2.flatten()
            RL2.store_transition(observation_p2_flatten, action_p2, reward_p2, observation_p1_flatten)
            if (step_p2 > 200) and (step_p2 % 5 == 0):
                RL2.learn(step_p2)
            step_p2 += 1

    print('game over')
    #########################################################
    #
    #     #print("observation_p1.shape, reward_p1, done_p1, info_p1:", observation_p1.shape, reward_p1, done_p1, info_p1)
    #
    #     #every move number
    #
    #
    #         # fresh env
    #
    #         # RL choose action based on observation
    #         observation_p1 = pre_process(observation_p1, crop_size)
    #         action_p2 = RL.choose_action(observation_p1)
    #        # print("action_p2:", action_p2)
    #         # RL take action and get next observation and reward
    #         draw_info_list = [game_number, reward_p2_total, step]
    #         observation_p2, reward_p2, done_p2, info_p2 = env.makeMove(player_p2, action_p2, draw_info_list, is_draw)
    #         reward_p2_total += reward_p2
    #         move_p2 += 1
    #         if done_p2:
    #             break
    #         observation_p1 = observation_p1.flatten()
    #         observation_p2 = observation_p2.flatten()
    #         RL2.store_transition(observation_p2, action_p2, reward_p2, observation_p1)
    #         if (step_p2 > 200) and (step_p2 % 5 == 0):
    #             RL.learn(step_p2)
    #         step_p2 += 1
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #         #step += 1
    #        # observation_, reward, done = env.step(action)
    #        # print("observation_p2, reward_p2, done_p2: ", observation_p2.shape, reward_p2, done_p2)
    #
    #         observation_p2 = pre_process(observation_p2, crop_size)
    #         observation_p1 = observation_p1.flatten()
    #         observation_p2 = observation_p2.flatten()
    #         RL.store_transition(observation_p1, action_p2, reward_p2, observation_p2)
    #
    #         if (step > 200) and (step % 5 == 0):
    #             RL.learn(step)
    #
    #         observation_p1, reward_p1, done_p1, info_p1 = env.run_player1(env)
    #
    #         # swap observation
    #
    #         # break while loop when end of this episode
    #         if done_p1:
    #             break
    #         step += 1
    #     print("game_number, reward_p2_total, reward_percent, step:", game_number, reward_p2_total, step)
    #        # env.draw_info(game_number, reward_p2_total, step)
    #
    # # end of game
    # print('game over')
    # #env.destroy()

def pre_process(frame, crop_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
    return frame

if __name__ == "__main__":
    # maze game
    env = Connect4()
    mode_type = 'train'

    #mode_type = 'use mode'
    RL1 = DeepQNetworkP1(mode=mode_type, n_actions=7, n_features=60*60*1,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    RL2 = DeepQNetworkP2(mode=mode_type, n_actions=7, n_features=60 * 60 * 1,
                         learning_rate=0.01,
                         reward_decay=0.9,
                         e_greedy=0.9,
                         replace_target_iter=200,
                         memory_size=2000,
                         output_graph=True
                         )
    if not os.path.exists('./checkpoints_p1'):
        os.makedirs('./checkpoints_p1')
    if not os.path.exists('./checkpoints_p2'):
        os.makedirs('./checkpoints_p2')
    run_connect4()

    RL1.plot_cost()
    RL2.plot_cost()
