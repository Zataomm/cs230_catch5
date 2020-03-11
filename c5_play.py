#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import numpy as np
import catch5_env
import c5utils
import c5ppo
import time
from tensorflow.keras.models import load_model

DEBUG = False # set to False if total_episodes is set to more than 1 


STATE_DIMS = (1,504)
N_ACTIONS = 64

winning_score = 31
tic = time.process_time()
total_time=0
win_total=[0,0]
total_hands=0
hand_win_total=[0,0]


TOTAL_GAMES=100
game_num=0

team_game_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]
num_hands_avg = [c5utils.RunningAvg()]

number_of_bids_won = [0,0,0,0]
average_winning_bid = [c5utils.RunningAvg(),c5utils.RunningAvg(),c5utils.RunningAvg(),c5utils.RunningAvg()]
average_rewards_per_winning_bid=[c5utils.RunningAvg(),c5utils.RunningAvg(),
                         c5utils.RunningAvg(),c5utils.RunningAvg()]

#setup policy network to use for playing for players #0 and 2 
_,_,policy = c5ppo.build_actor_critic_network(input_dims=STATE_DIMS, output_dims=N_ACTIONS)

#load networks from file
policy = load_model('models/policy_2200.hdf5')

#setup policy network to use for playing for players #0 and 2 
_,_,policy2 = c5ppo.build_actor_critic_network(input_dims=STATE_DIMS, output_dims=N_ACTIONS)

#load networks from file
policy2 = load_model('models/policy_1000.hdf5')


#setup environment
c5env=catch5_env.catch5()


while(game_num < TOTAL_GAMES):


    team_total = [0,0]
    num_hands=0  
    
    while team_total[0] < winning_score and team_total[1] < winning_score:

        if DEBUG:
            print("Dealer is player ",c5env.dealer)

        done = False
        while not done:
            observation = np.copy(c5env.states[c5env.current_player])
            int_obs =  np.copy(c5env.int_states[c5env.current_player])
            state_input = observation[np.newaxis,:]
            if DEBUG:
                c5utils.print_binstate(observation,c5env.current_player)

            legal_actions=c5env.legal_actions()

            if DEBUG:
                c5utils.print_actions(legal_actions)
            
            # now get the next move - and update states depending on who is playing
            if (c5env.current_player%2) == 0:
                action_dist = policy.predict([state_input], steps=1)
                legal_action_dist=c5env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
                #action = np.random.choice(N_ACTIONS, p=legal_action_dist[0, :])
                action = np.argmax(legal_action_dist[0, :])
                #force bid of only 3 
                #if action < 8:
                #    action = 1
            else:
                #action_dist = policy2.predict([state_input], steps=1)
                #legal_action_dist=c5env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
                #action = np.random.choice(N_ACTIONS, p=legal_action_dist[0, :])
                #action = np.argmax(legal_action_dist[0, :])
                action=c5utils.random_action(legal_actions)
                #force bid of only 3
                if action < 8:
                    action = 1 
                
            if DEBUG:
                c5utils.print_action(action)
                print("Step number = ",c5env.num_plays)
                
            #take a step and return next state 
            observation,reward,done,info = c5env.step(c5env.action_map[action])

        del observation,int_obs
        num_hands+=1

        number_of_bids_won[c5env.bidder]+=1
        average_winning_bid[c5env.bidder].set_avg(c5env.best_bid)
        average_rewards_per_winning_bid[c5env.bidder].set_avg(c5env.rewards[c5env.bidder])
        if DEBUG:
            c5utils.print_tricks(c5env.trick_info)

        if DEBUG:
            for i in range(4):
                print("Rewards for team ",i,":",c5env.rewards[i])

        team_total[0]+=c5env.rewards[0]
        team_total[1]+=c5env.rewards[1]

        if c5env.rewards[0] > c5env.rewards[1]:
            hand_win_total[0] += 1
        else:
            hand_win_total[1] += 1
        
        if DEBUG:
            print("====================  Scores team[0]:",team_total[0])
            print("====================  Scores team[1]:",team_total[1])
        
        
        #reset and count hands 
        c5env.reset()        
        
        if team_total[0]>=winning_score and team_total[0] > team_total[1]:
            win_total[0]+=1
        if team_total[1]>=winning_score and team_total[1] > team_total[0]:
            win_total[1] +=1

        
    num_hands_avg[0].set_avg(num_hands)
    team_game_avg[0].set_avg(team_total[0])
    team_game_avg[1].set_avg(team_total[1])
    
    game_num+=1


    print("Current score: Team 0:",win_total[0],"Team 1:",win_total[1])

for i in range(4):
    print("Player:",i,"number of winning bids:",number_of_bids_won[i])
    print("Average winning bid:",average_winning_bid[i].get_avg())
    print("Avg rewards per winning bid:",average_rewards_per_winning_bid[i].get_avg())

print("Wins for team 0:",win_total[0])
print("Wins for team 1:",win_total[1])
print("Hand won for team 0:",hand_win_total[0])
print("Hand won for team 1:",hand_win_total[1])
print("Point avg Team0:",team_game_avg[0].get_avg())
print("Point avg Team1:",team_game_avg[1].get_avg()) 
print("Average number of hands per game:",num_hands_avg[0].get_avg())
