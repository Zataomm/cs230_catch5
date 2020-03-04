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

from tensorflow.keras import backend as K


debug = True # set to False if total_runs is set to more than 1 
num_runs = 0
total_runs = 1 # set to 100000 to see averages for many runs 
winning_score = 31
total_time=0
win_total=[0,0]
total_hands=0

team_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]
team_bid_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]
team_game_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]

state_dims = (1,42)
n_actions = 64
 
model_actor = c5ppo.get_model_actor(input_dims=state_dims, output_dims=n_actions)
model_critic = c5ppo.get_model_critic(input_dims=state_dims)


tic = time.process_time()
while(num_runs < total_runs):
 
    test_c5=catch5_env.catch5()
    #store trajectories for the different players 
    # trajectories should be list of lists with [S,A,V,R,Done] for
    # the episode in time order ... needed for computing advantages
    trajectories=[[],[],[],[],[]]
    #for i in range(1):
    done = False
    while not done:
        observation = np.copy(test_c5.states[test_c5.current_player])
        state_input = K.eval(K.expand_dims(observation, 0))
        
        if debug:
            c5utils.print_state(observation,test_c5.current_player)
        legal_actions=test_c5.legal_actions(observation)
        print(test_c5.num_plays)
        if debug:
            c5utils.print_actions(legal_actions)
        #legal_actions=np.zeros((1,64))
        #legal_actions_input=K.eval(K.expand_dims(legal_actions, 0))
        action_dist = model_actor.predict([state_input], steps=1)
        legal_action_dist=test_c5.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
        #print(legal_action_dist)
        q_value = model_critic.predict([state_input], steps=1)
        #print("action_dist",action_dist,"shape",action_dist.shape,"q_val",q_value)
        action = np.random.choice(n_actions, p=legal_action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1           

        newtraj=[observation,action,np.squeeze(q_value),0,False]
        trajectories[test_c5.current_player].append(newtraj)
        if debug:
            c5utils.print_action(action)
            print("Step number = ",test_c5.num_plays)
        observation,reward,done,info = test_c5.step(test_c5.action_map[action])
        if done: #update last reward and done
            for i in range(4):
                trajectories[i][-1][4]=True
                trajectories[i][-1][3]=test_c5.rewards[i]             
        if debug:
            for i in range(4):
                c5utils.print_state(test_c5.states[(test_c5.current_player+i)%4],(test_c5.current_player+i)%4)
            c5utils.print_tricks(test_c5.trick_info)
            print(test_c5.rewards)
    if debug:
        for i in range(4):
            for j in range(len(trajectories[i])):
                c5utils.print_state(trajectories[i][j][0],i)
                print("Action:",trajectories[i][j][1],"Value:",
                      trajectories[i][j][2],"Reward:", trajectories[i][j][3],"Done:", trajectories[i][j][4])
    #print(trajectories)
        
    team_avg[0].set_avg(test_c5.game_points[0])
    team_avg[1].set_avg(test_c5.game_points[1])
    
    team_bid_avg[(test_c5.bidder)%2].set_avg(test_c5.rewards[(test_c5.bidder)%2])
    team_bid_avg[(test_c5.bidder+1)%2].set_avg(test_c5.rewards[(test_c5.bidder+1)%2])
  
    
    if test_c5.rewards[0] > test_c5.rewards[1]:
        win_total[0]+=1
    else:
        win_total[1]+=1
            
    test_c5.reset()    
    
    num_runs+=1
    total_hands=num_runs
    if (num_runs%100 == 0):
        toc = time.process_time()
        total_time = 1000*(toc-tic)
        print("Total runs so far:", num_runs)
        print("Total hands so far:",total_hands)
        print("Total time so far:",total_time,"ms")
        print("Avg time per game:",total_time/num_runs,"ms")
        print("Avg time per hand:",total_time/total_hands,"ms")
        print("Avg hands per game:",total_hands/num_runs)
        print("Team0 wins:",win_total[0],"Team1 wins:",win_total[1])
        print("Avg reward: Team 0:",team_avg[0].get_avg(),"Avg reward: Team 1:",team_avg[1].get_avg())
        print("Avg reward: Bidding:",team_bid_avg[0].get_avg(),"Avg reward: Non-bidding:",team_bid_avg[1].get_avg())
