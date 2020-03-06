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
from tensorflow.keras.callbacks import TensorBoard


DEBUG = False # set to False if total_episodes is set to more than 1 
EPISODE_NUM = 0
BATCH_SIZE=58
TOTAL_EPISODES = 100 # set to 1000 to see averages for many runs
STATE_DIMS = (1,42)
N_ACTIONS = 64
 
model_actor,model_critic,policy = c5ppo.build_actor_critic_network(input_dims=STATE_DIMS, output_dims=N_ACTIONS)

batch_states=[]
batch_actions=[]
batch_actions_onehot=[]
batch_prob=[]
batch_reward=[]
batch_value=[]
batch_returns=[]
batch_advantages=[]

tensor_board = TensorBoard(log_dir='./logs')

while(EPISODE_NUM < TOTAL_EPISODES):
 
    c5env=catch5_env.catch5()
    #store trajectories for the four different players 
    # trajectories should be list of lists with [S,A,V,R,Done] for
    # the episode in time order ... needed for computing advantages
    trajectories=[[],[],[],[],[]]
    #for i in range(1):
    done = False
    while not done:
        observation = np.copy(c5env.states[c5env.current_player])
        state_input = c5ppo.convert_state(observation)
        
        if DEBUG:
            c5utils.print_state(observation,c5env.current_player)
        legal_actions=c5env.legal_actions(observation)
        #print(c5env.num_plays)
        if DEBUG:
            c5utils.print_actions(legal_actions)
        action_dist = policy.predict([state_input], steps=1)
        legal_action_dist=c5env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
        q_value = model_critic.predict([state_input], steps=1)
        action = np.random.choice(N_ACTIONS, p=legal_action_dist[0, :])
        action_onehot = np.zeros(N_ACTIONS)
        action_onehot[action] = 1           
        newtraj=[observation,action,action_onehot,legal_action_dist,np.squeeze(q_value),0,False]
        trajectories[c5env.current_player].append(newtraj)
        if DEBUG:
            c5utils.print_action(action)
            print("Step number = ",c5env.num_plays)
        observation,reward,done,info = c5env.step(c5env.action_map[action])          
        if DEBUG:
            #for i in range(4):
            #    c5utils.print_state(c5env.states[(c5env.current_player+i)%4],(c5env.current_player+i)%4)
            c5utils.print_tricks(c5env.trick_info)
            print(c5env.rewards)
    # Now add the rewards,states, and values for terminal states i trajectories
    for i in range(4):
        observation = np.copy(c5env.states[i])
        state_input = c5ppo.convert_state(observation)
        q_value = 0
        newtraj=[observation,-1,None,None,q_value,c5env.rewards[i],True]
        trajectories[i].append(newtraj)      
    if DEBUG:
        for i in range(4):
            print("Length of trajectories:",i,len(trajectories[i]))
            for j in range(len(trajectories[i])):
                c5utils.print_state(trajectories[i][j][0],i)
                print("Action:",trajectories[i][j][1],"Value:",
                      trajectories[i][j][4],"Reward:", trajectories[i][j][5],"Done:", trajectories[i][j][6])
    eps_states=[]
    eps_actions=[]
    eps_actions_onehot=[]
    eps_prob=[]
    eps_reward=[]
    eps_value=[]
    eps_returns=[]
    eps_advantages=[] 

    for i in range(4):
        tmp_value=[]
        tmp_reward=[]
        for j in range(len(trajectories[i])-1):
            eps_states.append(trajectories[i][j][0])
            eps_actions.append(trajectories[i][j][1])
            eps_actions_onehot.append(trajectories[i][j][2])
            eps_prob.append(trajectories[i][j][3])
            eps_value.append(float(trajectories[i][j][4]))
            eps_reward.append(trajectories[i][j+1][5])
            tmp_value.append(float(trajectories[i][j][4]))
            tmp_reward.append(trajectories[i][j+1][5])
        player_returns,player_advantages= c5ppo.get_advantages(tmp_value,tmp_reward)
        eps_returns = eps_returns+player_returns
        eps_advantages = eps_advantages+player_advantages
    """
    print("States",len(eps_states))
    print(eps_states)
    print("Actions",len(eps_actions))
    print(eps_actions)
    print("Onehot",len(eps_actions_onehot))
    print(eps_actions_onehot)
    print("probs",len(eps_prob))
    print(eps_prob)
    print("Reward",len(eps_reward))
    print(eps_reward)
    print("Value",len(eps_value))
    print(eps_value)
    print("Returns",len(eps_returns))
    print(eps_returns)
    print("Adv",len(eps_advantages))
    print(eps_advantages)
    """

    batch_states=batch_states+eps_states
    batch_actions=batch_actions+eps_actions
    batch_actions_onehot=batch_actions_onehot+eps_actions_onehot
    batch_prob=batch_prob+eps_prob
    batch_reward=batch_reward+eps_reward
    batch_value=batch_value+eps_value
    batch_returns=batch_returns+eps_returns
    batch_advantages=batch_advantages+eps_advantages

    # reset and get next batch 
    c5env.reset()    
    EPISODE_NUM+=1

# now we have all of our data - lets train
batch_prob=np.asarray(batch_prob)
print(batch_prob.shape)
batch_advantages=np.reshape(batch_advantages,(len(batch_advantages),1,1))
batch_value=np.reshape(batch_value,(len(batch_value),1,1))

actor_loss = model_actor.fit(
    [batch_states, batch_prob, batch_advantages, np.reshape(batch_reward, newshape=(-1, 1, 1)), batch_value],
    [(np.reshape(batch_actions_onehot, newshape=(-1, N_ACTIONS)))],
    batch_size=BATCH_SIZE,verbose=True, shuffle=True, epochs=8,callbacks=[tensor_board])

critic_loss = model_critic.fit([batch_states], [np.reshape(batch_returns, newshape=(-1, 1))],
                               batch_size=BATCH_SIZE,shuffle=True, epochs=8,verbose=True, callbacks=[tensor_board])

    
