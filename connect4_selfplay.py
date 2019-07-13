# Author: Sascha Jecklin
# Date: 11/20/2018 15:22
# Connect Four Game in Python for Reinforcement Learning Project Thesis

import numpy as np
import sys
from scipy.ndimage.filters import convolve
from random import choice
from matplotlib import pyplot as plt
import pygame 
import random
# import matplotlib.image as maping
# import cv2
import subprocess
import glob
import os
import time 


class Connect4():
    def __init__(self, rowSize = 6, colSize= 7, winReward = 100, loseReward=-100, reward = 1):
        pygame.init()
        self.rowSize = rowSize
        self.colSize = colSize
        self.loseReward = loseReward
        self.winReward = winReward
        self.reward = reward
        self.field = np.zeros((self.rowSize, self.colSize))
        # self.field = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        self.turnCounter = 0
        self.N_games = 0
        self.P1_win = 0
        self.P1_lose = 0
        self.P2_win = 0
        self.P2_lose = 0
        self.P1_draw = 0
        self.P2_draw = 0 
        self.P1_info = []
        self.P2_info = []
        self.P1_illegal = 0 
        self.P2_illegal = 0
        self.P1_reward = 0
        self.P2_reward = 0
        self.win_reward = 1
        self.lose_reward = -1
        self.draw_reward = -1
        self.illegal_reward = -1
        # self.removeOldImages()
        # self.removeOldVideos()
        self.BLUE = (0, 0, 255)
        #self.LIGHT_BLUE = (0, 0, 128)
        self.BLACK = (255, 255, 255)
        self.WHITE = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
       # self.YELLOW = (255, 255, 0)
        # self.WHITE = (0,0,0)
        

        self.SQUARESIZE = 100
        self.WIDTH = self.colSize*self.SQUARESIZE
        self.HEIGHT = (1+self.rowSize)*self.SQUARESIZE
        self.SIZE = (self.WIDTH, self.HEIGHT)
        self.RADIUS = int(self.SQUARESIZE/2 - 5)
        self.SCREEN = pygame.display.set_mode(self.SIZE)

    def reset(self):
        self.field = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        self.turnCounter = 0
        self.P1_win = 0
        self.P1_lose = 0
        self.P2_win = 0
        self.P2_lose = 0
        self.P1_draw = 0
        self.P2_draw = 0 
        self.P1_info = []
        self.P2_info = []
        self.P1_illegal = 0 
        self.P2_illegal = 0
        return self.draw_board()  
        # stateSplitter()

    # def soft_reset(self):
    #     self.field = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
    #     return self.stateSplitter()

    # def getNextPlayer(self):
    #     if self.turnCounter % 2 == 0:
    #         return 1
    #     else:
    #         return 2
    
    def makeMove_P1(self, player, col, infoList, printflag): # returns -1 for error else observation, reward, done, info
        # pygame.init()
        reward = 0
        done = False
        self.P1_reward = 0
        self.P2_reward = 0

        self.draw_board()
        pygame.display.update()

        # assert self.turnCounter % 2 + 1 == player, "Not your turn Player: %r" % player
        selectedRow = self.field[:, col] #row to check
        if (selectedRow==0).any() == 0: 
            observation = self.image_data
            done = True
            #reward = self.illegal_reward
            self.P1_reward = self.illegal_reward
            self.P2_reward = self.win_reward
            self.P1_illegal = 1
            self.P2_win = 1
            self.P1_info = [self.P1_win, self.P1_draw, self.P1_lose, self.P1_illegal, col, self.field, self.P2_win, self.P2_draw, self.P2_lose, self.P2_illegal]
            if printflag == 1:
                print("player 1 run an illegal action")
            #return observation, reward, done, self.P1_info
            return observation, [self.P1_reward, self.P2_reward], done, self.P1_info
            # self.reset()
            # self.soft_reset()
            # if full return info
            # sys.exit('tried to fill an already full column')



        nextEmptySpace = (selectedRow!=0).argmin() #first nonzero entry starting from the bottom
        #nextEmptySpace = (selectedRow!=0).argmax() #first nonzero entry starting from the bottom
        self.field[nextEmptySpace, col] = player
        self.turnCounter += 1
        if printflag == 1:
            self.draw_board()
            # self.draw_info(infoList)
            # self.printField()

        # if imageflag == 1:
        #     self.makeImage(self.turnCounter)
        
        # observation = self.stateSplitter()
        observation = self.image_data
        winner = self.checkWin()  # check winner every played move
        if winner != 0:
            # self.makeVideo(self.turnCounter)
            if winner == 3:
                if printflag == 1:
                    print("It's a tie!")
                self.reset()
                done = True
                self.P1_draw = 1
                self.P2_draw = 1
                self.P1_reward = self.draw_reward
                self.P2_reward = self.draw_reward

            else:
                if winner == 1:
                    self.P1_reward = self.win_reward
                    self.P2_reward = self.lose_reward
                    self.P1_win = 1
                    self.P2_lose = 1
                else:
                    self.P1_reward = self.lose_reward
                    self.P2_reward = self.win_reward
                    self.P1_lose = 1
                    self.P2_win = 1

                if printflag == 1:
                    print("Player {} wins".format(winner))
                
                done = True
        self.P1_info = [self.P1_win, self.P1_draw, self.P1_lose, self.P1_illegal, col, self.field, self.P2_win, self.P2_draw, self.P2_lose, self.P2_illegal]
                
        return observation, [self.P1_reward, self.P2_reward], done, self.P1_info




    def makeMove_P2(self, player, col, infoList,printflag): # returns -1 for error else observation, reward, done, info
        # pygame.init()
        reward = 0
        done = False
        self.P1_reward = 0
        self.P2_reward = 0
        

        self.draw_board()
        pygame.display.update()


        # assert self.turnCounter % 2 + 1 == player, "Not your turn Player: %r" % player
        selectedRow = self.field[:, col] #row to check
        if (selectedRow==0).any() == 0: 
            observation = self.image_data
            done = True
            self.P2_reward = self.illegal_reward
            self.P1_reward = self.win_reward
            self.P2_illegal = 1
            self.P1_win = 1
            self.P2_info = [self.P1_win, self.P1_draw, self.P1_lose, self.P1_illegal, col, self.field, self.P2_win, self.P2_draw, self.P2_lose, self.P2_illegal]
            if printflag == 1:
                print("player 2 run an illegal action")
            return observation, [self.P1_reward, self.P2_reward], done, self.P2_info
           
            # self.soft_reset()
            # if full return info
            # sys.exit('tried to fill an already full column')



        nextEmptySpace = (selectedRow!=0).argmin() #first nonzero entry starting from the bottom
        #nextEmptySpace = (selectedRow!=0).argmax() #first nonzero entry starting from the bottom
        self.field[nextEmptySpace, col] = player
        self.turnCounter += 1
        if printflag == 1:
            self.draw_board()
            # self.draw_info(infoList)

            # self.printField()

        # if imageflag == 1:
        #     self.makeImage(self.turnCounter)
        
        # observation = self.stateSplitter()
        observation = self.image_data
        winner = self.checkWin() # check winner every played move
        if winner != 0:
            # self.makeVideo(self.turnCounter)
            if winner == 3:
                if printflag == 1:
                    print("It's a tie!")

                
                done = True
    
                self.P2_reward = self.draw_reward
                self.P1_reward = self.draw_reward
                self.P2_draw = 1
                self.P1_draw = 1

            else:
                if winner == 1:
                    self.P2_reward = self.lose_reward
                    self.P1_reward = self.win_reward
                    self.P2_lose = 1
                    self.P1_win = 1
                else:
                    self.P2_reward = self.win_reward
                    self.P1_reward = self.lose_reward
                    self.P2_win = 1
                    self.P1_lose = 1

                if printflag == 1:
                    print("Player {} wins".format(winner))
                    
                done = True
        #self.P2_info = [self.P2_win, self.P2_draw, self.P2_lose, self.P2_illegal, col, self.field]
        self.P2_info = [self.P1_win, self.P1_draw, self.P1_lose, self.P1_illegal, col, self.field, self.P2_win, self.P2_draw, self.P2_lose, self.P2_illegal]
        #print("self.p2_info:",self.P2_info)
        return observation, [self.P1_reward, self.P2_reward], done, self.P2_info

    # def stateSplitter(self): # makes a matrix for player 1 and one for palyer 2
    #     copy_player_1 = np.copy(self.field)
    #     copy_player_2 = np.copy(self.field)
    #     copy_player_1[copy_player_1 == 2] = 0
    #     copy_player_2[copy_player_2 == 1] = 0
    #     copy_player_2[copy_player_2 == 2] = 1
    #     return np.stack([copy_player_1, copy_player_2], axis = -1)

    def draw_board(self):
        newfield = np.rot90(self.field,-1)
        # data = np.zeros((self.rowSize, self.colSize))
        # #data = data*255 #white background
        # data = np.flip(data, 0)
        
        for c in range(self.colSize):
            for r in range(self.rowSize):
                loc_size = (c*self.SQUARESIZE, (r+1)*self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE)
                pygame.draw.rect(self.SCREEN, self.BLUE, loc_size)
                #pygame.draw.rect(self.SCREEN, self.WHITE, loc_size)
                loc = (int((c+0.5)*self.SQUARESIZE), int((r+1.5)*self.SQUARESIZE))
                pygame.draw.circle(self.SCREEN, self.BLACK, loc, self.RADIUS)

        for c in range(self.colSize):
            for r in range(0, self.rowSize):		
                if newfield[c][r] == 1:
                    loc = (int((c+0.5)*self.SQUARESIZE),int((r+1.5)*self.SQUARESIZE) )
                    pygame.draw.circle(self.SCREEN, self.RED, loc, self.RADIUS)
                elif newfield[c][r] == 2: 
                    loc = (int((c+0.5)*self.SQUARESIZE),int((r+1.5)*self.SQUARESIZE) )
                    pygame.draw.circle(self.SCREEN, self.GREEN, loc, self.RADIUS)

        self.image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # print("----------------------/n")
        # print(image_data)
        pygame.display.update()
        return (self.image_data)

    # def draw_info(self,infoList):
    #     pygame.font.init()
    #     myfont = pygame.font.Font(None, 35)

    #     label_GameNumber = myfont.render("Game_Number:" + str(infoList[0]), 1, self.BLACK)
    #     self.SCREEN.blit(label_GameNumber,(30,30))

    #     label_P2_score = myfont.render("score:" + str(infoList[1]),1,self.BLACK)
    #     self.SCREEN.blit(label_P2_score,(30,50))

    #     label_p2_win = myfont.render("P2_win:" + str(self.P2_win),1,self.BLACK)
    #     self.SCREEN.blit(label_p2_win,(300,30))

    #     label_step = myfont.render("step:" + str(infoList[2]),1,self.BLACK)
    #     self.SCREEN.blit(label_step,(300,50))


    # def printField(self):
    #     for i in range(np.size(self.field, 0)-1,-1,-1):
    #         for j in range(0, self.rowSize+1):
    #             if self.field[i,j] == 1:
    #                 print('\x1b[0;30;43m' + str(self.field[i,j]) + '\x1b[0m', end='  ')
    #             elif self.field[i,j] == 2:
    #                 print('\x1b[0;30;41m' + str(self.field[i,j]) + '\x1b[0m', end='  ') #1;30;43m in spyder
    #             else:
    #                 print(self.field[i,j], end='  ')
    #         print('\n')
    #     print("\n")


    def checkWin(self): # Convolves pattern with field. If a 4 appears --> winner
        mask = np.zeros(shape=(self.rowSize, self.colSize), dtype=int)
        if self.turnCounter % 2:
             mask[self.field==1]=1
             possbileWinner = 1
        else:
             mask[self.field==2]=1
             possbileWinner = 2
        if self.turnCounter == self.colSize * self.rowSize:
            return 3
        k1 = np.array([[1],[1],[1],[1]])
        k2 = np.array([[1,1,1,1]])
        k3 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        k4 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])

        convVertical = convolve(mask, k1, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k2, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k3, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner
        convVertical = convolve(mask, k4, mode='constant', cval=0)
        if 4 in convVertical:
            return possbileWinner

        return 0

    # def validMoves(self): #returns an array colSize long with ones for validMoves
    #     validMovesArray = np.ones(self.colSize,dtype = int)
    #     for i in range(0, self.rowSize+1):
    #         selectedRow = self.field[:, i] #row to check
    #         if (selectedRow==0).any() == 0: # if full return -1
    #             validMovesArray[i] = 0
    #     return validMovesArray

    # def sample(self): #returns a valid sample move
    #     return choice(np.where(self.validMoves())[0]) # [0] to get it out of the tuple

    # def makeImage(self, i): #makes an image from the current state
    #     data = np.zeros((self.rowSize, self.colSize, 3), dtype=np.uint8)
    #     #data = data*255 #white background
    #     data[self.field == 1] = [255, 240, 0]
    #     data[self.field == 2] = [255, 0, 0]
    #     data = np.flip(data, 0)
    #     plt.figure()
    #     print("--------------------\n")
    #     print(data.shape)
    #     print(data)
    #     im = plt.imshow(data, interpolation='none', vmin=0, vmax=1, aspect='equal');
    #     ax = plt.gca();
    #     ax.set_xticks(np.arange(-.5, self.colSize, 1), minor=True); #need to shift grid to right position
    #     ax.set_yticks(np.arange(-.5, self.rowSize, 1), minor=True);
    #     plt.tick_params(
    #         axis='both',       # changes apply to the x and y-axis
    #         which='both',      # both major and minor ticks are affected
    #         bottom=False,      # ticks along the bottom edge are off
    #         top=False,
    #         left=False,
    #         right=False,       # ticks along the top edge are off
    #         labelleft=False,
    #         labelbottom=False) # labels along the bottom edge are off

    #     ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    #     plt.savefig("/Users/kurt/Game/Project/IMG/move%02d.png" % i, bbox_inches='tight', pad_inches = 0)
    #     # G = cv2.imread("/Users/kurt/Game/Project/IMG/move%02d.png" % i)
    #     # print("************************/n")
    #     # print(G.shape)
    #     # print(G)
       

    #     plt.close()


    # def makeVideo(self, i): #makes a video out of saved images
    #     #print("Saving Video...")
    #     FNULL = open(os.devnull, 'w') #subprocess ouput redirected to devnull
    #     subprocess.call([
    #         'ffmpeg', '-framerate', '2', '-i', 'move%02d.png', '-r', '2', '-pix_fmt', 'yuv420p',
    #         "game%02d.mp4" % i
    #     ], stdout=FNULL, stderr=subprocess.STDOUT)
    #     self.removeOldImages()

    # def removeOldImages(self):
    #     for file_name in glob.glob("*.png"):
    #         os.remove(file_name)

    # def removeOldVideos(self):
    #     for file_name in glob.glob("*.mp4"):
    #         os.remove(file_name)


    def run_player1(self, game):

        info = 1
        while info != None:
            value = [0,1,2,3,4,5,6]
            # print(game.validMoves())
            
            userInput = random.choice(value)
            # userInput = int(input("Which row Player 1? "))
            # time.sleep(3)
            observation, reward, done, info = game.makeMove_P1(1,userInput,[], 1,)
            return observation, reward, done, info
            # print(observation)
            # if info != None:
            #     print(info)
                
        # info = 1
        # while info != None:
        #     # print(game.validMoves())
            
        #     # userInput = rand.choice(value)
            
        #     userInput = int(input("Which row Player 2?"))
        #     observation, reward, done, info = game.makeMove(2,userInput, 1)
        #     print(observation)
        #     # if info != None:
        #     #     print(info)
                   
