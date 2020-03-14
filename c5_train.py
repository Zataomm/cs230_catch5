#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import matplotlib.pyplot as plt
import argparse
import numpy as np
import catch5_env
import c5utils
import c5ppo
import time
from tensorflow.keras.callbacks import TensorBoard,LearningRateScheduler
from tensorflow.keras.models import save_model

# set up argument parser for parameters for training
parser = argparse.ArgumentParser(description='Parser for catch5 training script')

parser.add_argument('-epochs', action='store',
                    type=int,
                    default=5,
                    dest='epochs',
                    help='Number of epochs (passes through the data) per training iteration')
parser.add_argument('-iters', action='store',
                    type=int,
                    default=10000001,
                    dest='iterations',
                    help='Number of training iterations (generation of batchs of data) to execute')
parser.add_argument('-batch', action='store',
                    type=int,
                    default=8,
                    dest='batch_size',
                    help='Size of mini-batch to use for training')

parser.add_argument('-debug', action='store_true',
                    default=False,
                    dest='debug',
                    help='Turn on debugging')

parser.add_argument('-plot', action='store_true',
                    default=False,
                    dest='plot',
                    help='Turn on plotting for the loss functions')

parser.add_argument('-eb', action='store',
                    type=float,
                    default=0.01,
                    dest='entropy_beta',
                    help='Entropy beta to the loss function')

parser.add_argument('-lr', action='store',
                    type=float,
                    default=0.000025,
                    dest='learning_rate',
                    help='Learning rate for the algorithm')

parser.add_argument('-episodes', action='store',
                    type=int,
                    default=32,
                    dest='episodes',
                    help='Number of full episodes to run for each training batch.(1 eps = 29 trjectories)')

parser.add_argument('-save', action='store',
                    type=int,
                    default=500,
                    dest='save_every',
                    help='Save weights after given iterations')


class run_training():
    """ Class used to train networks to learn how to play the game catch5.  
    """
    def __init__(self,DEBUG=False,CLIP_VAL=0.2,CRITIC_DIS=0.5,ENTROPY_BETA=0.01,GAMMA=0.99,
                 LMBDA=0.95,LR=0.00005,BATCH_SIZE=8,EPOCHS=5,TOTAL_EPISODES=32,STATE_DIMS=504,
                 N_ACTIONS=64,ITERATIONS=1000001,SAVE_EVERY=50):

        #parameters
        self.clipping_val = CLIP_VAL
        self.critic_discount = CRITIC_DIS
        self.entropy_beta = ENTROPY_BETA
        self.gamma = GAMMA
        self.lmbda = LMBDA
        self.learning_rate = LR
        self.DEBUG = DEBUG 
        self.BATCH_SIZE=BATCH_SIZE
        self.EPOCHS=EPOCHS
        self.TOTAL_EPISODES = TOTAL_EPISODES # x29  = number of trajectories
        self.STATE_DIMS = STATE_DIMS
        self.N_ACTIONS = N_ACTIONS
        self.SAVE_EVERY=SAVE_EVERY

        self.batch_states=[]
        self.batch_actions=[]
        self.batch_actions_onehot=[]
        self.batch_prob=[]
        self.batch_reward=[]
        self.batch_value=[]
        self.batch_returns=[]
        self.batch_advantages=[]

        self.model_actor,self.policy = c5ppo.build_actor_network(input_dims=self.STATE_DIMS,output_dims=self.N_ACTIONS,
                            learning_rate=self.learning_rate,clipping_val=self.clipping_val,entropy_beta=self.entropy_beta)
        self.model_critic = c5ppo.build_critic_network(input_dims=self.STATE_DIMS,learning_rate=self.learning_rate)
        self.tensor_board = TensorBoard(log_dir='./logs')


    
    def lr_scheduler(epoch, lr):
        decay_rate = 0.85
        decay_step = 1
        if epoch % decay_step == 0 and epoch:
            return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr       

    def save_models(self,name):
        print("Saving network weights with file extension:",name)
        self.model_actor.save_weights('models/model_actor_{}.hdf5'.format(name))
        self.model_critic.save_weights('models/model_critic_{}.hdf5'.format(name))
        self.policy.save_weights('models/policy_{}.hdf5'.format(name))


    def generate_batch(self):

        self.save_models('init')

        #reset batch states to be empty
        self.batch_states=[]
        self.batch_actions=[]
        self.batch_actions_onehot=[]
        self.batch_prob=[]
        self.batch_reward=[]
        self.batch_value=[]
        self.batch_returns=[]
        self.batch_advantages=[]

        episode_num=0
        
        while(episode_num < self.TOTAL_EPISODES):

            c5env=catch5_env.catch5()
            #store trajectories for the four different players 
            # trajectories should be list of lists with [S,A,V,R,Done] for
            # the episode in time order ... needed for computing advantages
            trajectories=[[],[],[],[],[]]
            done = False
            while not done:
                observation = np.copy(c5env.states[c5env.current_player])
                int_obs =  np.copy(c5env.int_states[c5env.current_player])
                state_input = observation[np.newaxis,:]

                if self.DEBUG:
                    c5utils.print_intstate(int_obs,c5env.current_player)
                    c5utils.print_binstate(observation,c5env.current_player)               
                legal_actions=c5env.legal_actions()
                if self.DEBUG:
                    c5utils.print_actions(legal_actions)
                action_dist = self.policy.predict([state_input], steps=1)
                legal_action_dist=c5env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
                q_value = self.model_critic.predict([state_input], steps=1)
                action = np.random.choice(self.N_ACTIONS, p=legal_action_dist[:])
                action_onehot = np.zeros(self.N_ACTIONS)
                action_onehot[action] = 1           
                newtraj=[observation,action,action_onehot,legal_action_dist,np.squeeze(q_value),0,False]
                trajectories[c5env.current_player].append(newtraj)
                if self.DEBUG:
                    c5utils.print_action(action)
                    print("Step number = ",c5env.num_plays)
                observation,reward,done,info = c5env.step(c5env.action_map[action])
                if self.DEBUG:
                    c5utils.print_tricks(c5env.trick_info)
                    print(c5env.rewards)
            # Now add the rewards,states, and values for terminal states i trajectories
            for i in range(4):
                observation = np.copy(c5env.states[i])
                state_input = observation[np.newaxis,:]
                q_value = 0
                newtraj=[observation,-1,None,None,q_value,c5env.rewards[i]/9.0,True]
                trajectories[i].append(newtraj)      
            if self.DEBUG:
                for i in range(4):
                    print("Length of trajectories:",i,len(trajectories[i]))
                    for j in range(len(trajectories[i])):
                        c5utils.print_binstate(trajectories[i][j][0],i)
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
                player_returns,player_advantages= c5ppo.get_advantages(tmp_value,tmp_reward,self.gamma,self.lmbda)
                eps_returns = eps_returns+player_returns
                eps_advantages = eps_advantages+player_advantages

            if self.DEBUG:
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


            self.batch_states=self.batch_states+eps_states
            self.batch_actions=self.batch_actions+eps_actions
            self.batch_actions_onehot=self.batch_actions_onehot+eps_actions_onehot
            self.batch_prob=self.batch_prob+eps_prob
            self.batch_reward=self.batch_reward+eps_reward
            self.batch_value=self.batch_value+eps_value
            self.batch_returns=self.batch_returns+eps_returns
            self.batch_advantages=self.batch_advantages+eps_advantages

            # reset and get next batch
            episode_num+=1
            c5env.reset()    


    def reformat_batch(self):
        # now we have all of our data - format and train
        self.batch_prob=np.asarray(self.batch_prob)
        bln=len(self.batch_advantages)
        self.batch_prob=np.reshape(self.batch_prob,newshape=(-1, self.N_ACTIONS))
        self.batch_advantages=np.asarray(self.batch_advantages)
        self.batch_advantages= (self.batch_advantages-self.batch_advantages.mean()) / (self.batch_advantages.std() + 1e-8)
        self.batch_advantages=np.reshape(self.batch_advantages,(bln))
        self.batch_value=np.asarray(self.batch_value)
        self.batch_value=np.reshape(self.batch_value,(len(self.batch_value)))
        
        self.batch_states=np.asarray(self.batch_states)
        self.batch_reward=np.asarray(self.batch_reward)
        self.batch_actions_onehot=np.asarray(self.batch_actions_onehot)

        self.batch_reward =np.reshape(self.batch_reward, newshape=(-1))
        self.batch_actions_onehot=np.reshape(self.batch_actions_onehot, newshape=(-1, self.N_ACTIONS))

        self.batch_returns=np.asarray(self.batch_returns)
        self.batch_returns=np.reshape(self.batch_returns, newshape=(-1))

        if self.DEBUG:
            print("Shapes going into training:")
            print("\t batch_states:",self.batch_states.shape)
            print("\t batch_prob:",self.batch_prob.shape)
            print("\t batch_advantages:",self.batch_advantages.shape)
            print("\t batch_reward:",self.batch_reward.shape)  
            print("\t batch_value:",self.batch_value.shape)
            print("\t batch_onehot:",self.batch_actions_onehot.shape)
            print("\t batch_returns:",self.batch_returns.shape)
        
            
    def compute_grads(self,itrs):
   
        self.reformat_batch()

        if self.DEBUG:
            print("Shapes going into training:")
            print("\t batch_states:",self.batch_states.shape)
            print("\t batch_prob:",self.batch_prob.shape)
            print("\t batch_advantages:",self.batch_advantages.shape)
            print("\t batch_reward:",self.batch_reward.shape)  
            print("\t batch_value:",self.batch_value.shape)
            print("\t batch_onehot:",self.batch_actions_onehot.shape)
            print("\t batch_returns:",self.batch_returns.shape)

        
        actor_loss = self.model_actor.fit([self.batch_states, self.batch_prob, self.batch_advantages],
                                          [self.batch_actions_onehot],batch_size=self.BATCH_SIZE,verbose=True, shuffle=True,
                                          epochs=self.EPOCHS,callbacks=[self.tensor_board])

        
        critic_loss = self.model_critic.fit([self.batch_states], [self.batch_returns],batch_size=self.BATCH_SIZE,shuffle=True,
                                            epochs=self.EPOCHS,verbose=True, callbacks=[self.tensor_board])
        
        if itrs%self.SAVE_EVERY == 0:
            self.save_models(itrs)
                
        return actor_loss,critic_loss

if __name__ == "__main__":

    args = parser.parse_args()

    print("Batch size:",args.batch_size)
        

    train = run_training(EPOCHS=args.epochs,BATCH_SIZE=args.batch_size,DEBUG=args.debug,ENTROPY_BETA=args.entropy_beta,
                         LR=args.learning_rate,TOTAL_EPISODES=args.episodes,SAVE_EVERY=args.save_every)

    actor_loss = []
    critic_loss= []
    
    for i in range(args.iterations):
        train.generate_batch()
        a_hist,c_hist  = train.compute_grads(0)
        print("\na_hist dict:",a_hist.history)
        print("\nc_hist dict:",c_hist.history)
        print(np.mean(np.asarray(a_hist.history['loss'])),np.mean(np.asarray(c_hist.history['loss'])))
        actor_loss =  actor_loss+[np.mean(np.asarray(a_hist.history['loss']))]
        critic_loss = critic_loss+[np.mean(np.asarray(c_hist.history['loss']))]
        if args.plot:
            x_axis=range(i+1)
            plt.cla()
            plt.grid(True)
            plt.ylabel("actor, critic loss")
            plt.xlabel("Number of iterations")
            plt.title("Actor(b)-Critic(g) Loss vs Iterations")
            plt.plot(x_axis,actor_loss,'b',x_axis,critic_loss,'g')
            plt.draw()
            plt.pause(0.01)
    
