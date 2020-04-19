#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import matplotlib.pyplot as plt
from itertools import permutations 
import argparse
import numpy as np
import catch5_env
import c5utils
import c5ppo
import time
from tensorflow.keras.callbacks import TensorBoard,LearningRateScheduler
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

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

parser.add_argument('-alr', action='store',
                    type=float,
                    default=0.000025,
                    dest='a_lr',
                    help='Learning rate for the actor network')

parser.add_argument('-clr', action='store',
                    type=float,
                    default=0.000025,
                    dest='c_lr',
                    help='Learning rate for the critic network')

parser.add_argument('-clip', action='store',
                    type=float,
                    default=0.2,
                    dest='clip_val',
                    help='Clipping Value for PPO')

parser.add_argument('-episodes', action='store',
                    type=int,
                    default=32,
                    dest='episodes',
                    help='Number of full episodes to run for each training batch.(1 eps = 29 trajectories)')

parser.add_argument('-save', action='store',
                    type=int,
                    default=500,
                    dest='save_every',
                    help='Save weights after given iterations')

parser.add_argument('-intstate', action='store_true',
                    default=False,
                    dest='intstate',
                    help='Train with 42 dimensional integer states instead of default 504 dimensional binary states')

parser.add_argument('-state_dims', action='store',
                    type=int,
                    default=504,
                    dest='state_dims',
                    help='Input state dimensions for NN - set to 42 if using -intstate option.')

parser.add_argument('-numperms', action='store',
                    type=int,
                    default=-1,
                    dest='num_perms',
                    help='Number of permutations to augment the data from the current trajectories.')
parser.add_argument('-activation', action='store',
                    type=str,
                    default="tanh",
                    dest='act_type',
                    help='Default is tanh, use \'leaky\' for Leay ReLU activations.')
parser.add_argument('-start_iteration', action='store',
                    type=int,
                    default=0,
                    dest='start_iter',
                    help='Load a set of weights at the start of training based on iteration number. There needs to be a set of weight files in the models directory with the appropriate iteration number. e.g. if iteration number is 1000, there should be a model_actor_1000.hdf5, model_critic_1000.hdf5 and a policy_1000.hdf5 in the models directory. Training will start from this file.')


class run_training():
    """ Class used to train networks to learn how to play the game catch5.  
    """
    def __init__(self,DEBUG=False,CLIP_VAL=0.2,ENTROPY_BETA=0.01,GAMMA=0.99,
                 LMBDA=0.95,C_LR=0.00005,A_LR=0.00005,BATCH_SIZE=8,EPOCHS=5,TOTAL_EPISODES=32,STATE_DIMS=504,
                 N_ACTIONS=64,ITERATIONS=1000001,SAVE_EVERY=50,USE_INT_STATES=False,NUM_PERMS=-1,ACT_TYPE="tanh",START_ITER=0):

        #parameters
        self.clipping_val = CLIP_VAL
        self.entropy_beta = ENTROPY_BETA
        self.gamma = GAMMA
        self.lmbda = LMBDA
        self.actor_learning_rate = A_LR
        self.critic_learning_rate = C_LR
        self.min_lr = 0.00001
        self.DEBUG = DEBUG 
        self.BATCH_SIZE=BATCH_SIZE
        self.EPOCHS=EPOCHS
        self.TOTAL_EPISODES = TOTAL_EPISODES # x29  = number of trajectories
        self.STATE_DIMS = STATE_DIMS
        self.N_ACTIONS = N_ACTIONS
        self.SAVE_EVERY=SAVE_EVERY
        self.USE_INT_STATES=USE_INT_STATES
        self.num_perms=NUM_PERMS
        self.act_type=ACT_TYPE

        self.suit_perms=list(permutations(range(4))) 
        self.batch_states=[]
        self.batch_actions=[]
        self.batch_actions_onehot=[]
        self.batch_prob=[]
        self.batch_reward=[]
        self.batch_value=[]
        self.batch_returns=[]
        self.batch_advantages=[]

        self.mass_beta = 0.999
        self.avg_mass = 0.0
        self.avg_zerop = 0.0

        self.bid_metric = 0.5
        self.bid_beta = 0.99

        self.bid_suits = [0,0,0,0]

        self.reward_norm = 9.0 # should be max reward ....

        self.model_actor,self.policy = c5ppo.build_actor_network(input_dims=self.STATE_DIMS,output_dims=self.N_ACTIONS,
                                                                 learning_rate=self.actor_learning_rate,clipping_val=self.clipping_val,entropy_beta=self.entropy_beta,act_type=self.act_type)
        self.model_critic = c5ppo.build_critic_network(input_dims=self.STATE_DIMS,learning_rate=self.critic_learning_rate,act_type=self.act_type)
        self.tensor_board = TensorBoard(log_dir='./logs')

        self.start_iter=START_ITER


        print("Starting training from iteration",self.start_iter)
        if self.start_iter == 0:
            self.save_models('init')
        elif self.start_iter > 0:
            self.load_models(str(self.start_iter))
        

    def load_models(self,name):
        print("Loading network weights with file extension:",name)
        self.model_actor.load_weights('models/model_actor_{}.hdf5'.format(name))
        self.model_critic.load_weights('models/model_critic_{}.hdf5'.format(name))
        self.policy.load_weights('models/policy_{}.hdf5'.format(name))

    def save_models(self,name):
        print("Saving network weights with file extension:",name)
        self.model_actor.save_weights('models/model_actor_{}.hdf5'.format(name))
        self.model_critic.save_weights('models/model_critic_{}.hdf5'.format(name))
        self.policy.save_weights('models/policy_{}.hdf5'.format(name))


    def step_decay(self,curriter):
        drop = 0.5
        curriter_drop = 10.0
        lrate = self.critic_learning_rate * np.power(drop,np.floor((1+curriter)/curriter_drop))
        return min(lrate,self.min_lr)

    def exp_decay(self,curriter,k):
        lrate = self.critic_learning_rate * np.exp(-k*curriter)
        return min(lrate,self.min_lr)

    def time_decay(self,curriter,k):
        lrate = self.critic_learning_rate/(1+k*curriter)
        return min(lrate,self.min_lr)
    
    def generate_batch(self):

        #reset batch states to be empty
        self.batch_states=[]
        self.batch_actions=[]
        self.batch_actions_onehot=[]
        self.batch_prob=[]
        self.batch_reward=[]
        self.batch_value=[]
        self.batch_returns=[]
        self.batch_advantages=[]
        self.bid_suits = [0,0,0,0]
        episode_num=0
        
        while(episode_num < self.TOTAL_EPISODES):

            c5env=catch5_env.catch5()
            #store trajectories for the four different players 
            # trajectories should be list of lists with [S,A,V,R,Done] for
            # the episode in time order ... needed for computing advantages
            trajectories=[[],[],[],[]]
            done = False
            while not done:
                observation = np.copy(c5env.states[c5env.current_player])
                int_obs =  np.copy(c5env.int_states[c5env.current_player])
                if not self.USE_INT_STATES:
                    state_input = observation[np.newaxis,:]
                else:
                    state_input = int_obs[np.newaxis,:]
                if self.DEBUG:
                    c5utils.print_intstate(int_obs,c5env.current_player)
                    c5utils.print_binstate(observation,c5env.current_player)               
                legal_actions=c5env.legal_actions()
                if self.DEBUG:
                    c5utils.print_actions(legal_actions)
                action_dist = self.policy.predict([state_input], steps=1)
                legal_action_dist,pmass,num_moves,num_zerop_moves=c5env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
                self.avg_mass = pmass + self.mass_beta*(self.avg_mass-pmass)
                self.avg_zerop = num_zerop_moves+self.mass_beta*(self.avg_zerop-num_zerop_moves)
                q_value = self.model_critic.predict([state_input], steps=1)
                action = np.random.choice(self.N_ACTIONS, p=legal_action_dist[:])
                action_onehot = np.zeros(self.N_ACTIONS)
                action_onehot[action] = 1
                if not self.USE_INT_STATES:
                    newtraj=[observation,action,action_onehot,legal_action_dist,np.squeeze(q_value),0,False]
                else:
                    newtraj=[int_obs,action,action_onehot,legal_action_dist,np.squeeze(q_value),0,False]
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
                newtraj=[observation,-1,None,None,q_value,c5env.rewards[i]/self.reward_norm,True]
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
            curr_bm = c5env.bid_max_cards+c5env.bid_max_value
            self.bid_metric = curr_bm + self.bid_beta*(self.bid_metric-curr_bm)
            self.bid_suits[int(c5env.bid_suit)] += 1
            episode_num+=1
            c5env.reset()    


    def augment_data(self):
        """ For every given state we can generate 23 (4!-1) new states for training - which will ensure the 
            more diverse data for training to help with suit selection and bidding etc.... Instead of looking at random perms 
            we will look at perms that ensure that we mixt the suit selection up - so bids that augment the trajectories so
            that we have the bidding suit show up as all other possible suits. So we keep the original trajectory - and add three others.  
        """
        
        all_rand_perms=[[(0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2)],
                    [(0,1,2,3),(1,0,3,2),(2,3,0,1),(3,2,1,0)],
                    [(0,1,2,3),(1,0,3,2),(2,3,1,0),(3,2,0,1)],
                    [(0,1,2,3),(1,3,0,2),(2,0,3,1),(3,2,1,0)]]
        rand_perms=all_rand_perms[np.random.randint(4)]
        
        self.batch_actions=len(rand_perms)*self.batch_actions
        self.batch_reward=len(rand_perms)*self.batch_reward
        self.batch_value=len(rand_perms)*self.batch_value
        self.batch_returns=len(rand_perms)*self.batch_returns
        self.batch_advantages=len(rand_perms)*self.batch_advantages    
        new_bstates=[]
        new_baoh=[]
        new_bprobs=[]
        for p in rand_perms:
            for indx in range(len(self.batch_states)):
                s=self.batch_states[indx]
                aoh=self.batch_actions_onehot[indx]
                prob=self.batch_prob[indx]
                new_s=np.copy(s)
                new_aoh=np.copy(aoh)
                new_prob=np.copy(prob)
                #permute the suits
                for i in range(4):
                    new_s[32+i]=s[32+p[i]]
                    new_aoh[8+i] = aoh[8+p[i]]
                    new_prob[8+i] = prob[8+p[i]]
                #permute aoh and probs
                for k in range(4):
                    new_aoh[12+13*k:12+13*(k+1)]=aoh[12+13*p[k]:12+13*(p[k]+1)]
                    new_prob[12+13*k:12+13*(k+1)]=prob[12+13*p[k]:12+13*(p[k]+1)]
                if self.DEBUG:
                    print("\n aoh ====================== \n",aoh)
                    print("\n new aoh ====================== \n",new_aoh)
                    print("\n prob ====================== \n",prob)
                    print("\n new prob ====================== \n",new_prob)
                #permute the 9 decks in state suits as well
                for j in range(9):
                    for k in range(4):
                        new_s[36+52*j+13*k:36+52*j+13*(k+1)]=s[36+52*j+13*p[k]:36+52*j+13*(p[k]+1)]
                if self.DEBUG:
                    for i in range(4):
                        print(p)
                        print("New state      ====================")
                        c5utils.print_binstate(new_s,i)
                        print("Original state ====================")
                        c5utils.print_binstate(s,i)
                new_bstates.append(new_s)
                new_baoh.append(new_aoh)
                new_bprobs.append(new_prob)
        self.batch_states=new_bstates
        self.batch_actions_onehot=new_baoh
        self.batch_prob = new_bprobs
        
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

        if self.num_perms > 0:
            self.augment_data()
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
                                          epochs=self.EPOCHS)

        
        critic_loss = self.model_critic.fit([self.batch_states], [self.batch_returns],batch_size=self.BATCH_SIZE,shuffle=True,
                                            epochs=self.EPOCHS,verbose=True)
        
        if itrs%self.SAVE_EVERY == 0:
            self.save_models(itrs)
                
        return actor_loss,critic_loss

if __name__ == "__main__":

    args = parser.parse_args()

    print("Batch size:",args.batch_size)
        

    train = run_training(EPOCHS=args.epochs,CLIP_VAL=args.clip_val,BATCH_SIZE=args.batch_size,DEBUG=args.debug,ENTROPY_BETA=args.entropy_beta,
                         C_LR=args.c_lr,A_LR=args.a_lr, TOTAL_EPISODES=args.episodes,SAVE_EVERY=args.save_every,
                         USE_INT_STATES=args.intstate,STATE_DIMS=args.state_dims,NUM_PERMS=args.num_perms,ACT_TYPE=args.act_type,START_ITER=args.start_iter)

    actor_loss = []
    critic_loss= []
    prob_mass = []
    bid_metric = []
    bid_suits = np.zeros(4)

    if args.plot:
        bfig = plt.figure()
        bfig.suptitle('Actor-Critic Loss/Probability Mass/Bid Metric', fontsize=12)
        bax1 = bfig.add_subplot(411)
        bax2 = bfig.add_subplot(412)
        bax3 = bfig.add_subplot(413)
        bax4 = bfig.add_subplot(414)
        
    
    for i in range(args.start_iter,args.iterations):
        train.generate_batch()
        a_hist,c_hist  = train.compute_grads(i)
        print("\na_hist dict:",a_hist.history)
        print("\nc_hist dict:",c_hist.history)
        print(np.mean(np.asarray(a_hist.history['loss'])),np.mean(np.asarray(c_hist.history['loss'])))
        actor_loss =  actor_loss+[np.mean(np.asarray(a_hist.history['loss']))]
        critic_loss = critic_loss+[np.mean(np.asarray(c_hist.history['loss']))]
        prob_mass = prob_mass + [train.avg_mass]
        bid_metric =  bid_metric + [train.bid_metric]
        bid_suits = bid_suits+np.asarray(train.bid_suits)
        bid_averages = np.copy(bid_suits)/((i+1)*args.episodes)
        bid_averages_last = np.asarray(train.bid_suits)/args.episodes
        print("Average mass at iteration",i," = ",train.avg_mass)
        print("Average zero probability moves at iteration",i," = ",train.avg_zerop)
        print("Bid metric at iteration",i," = ",train.bid_metric)
        print("Clubs:",bid_averages[0],"Diamonds:",bid_averages[1],"Hearts:",bid_averages[2],"Spades:",bid_averages[3])
        print("Clubs:",bid_averages_last[0],"Diamonds:",bid_averages_last[1],"Hearts:",bid_averages_last[2],"Spades:",bid_averages_last[3])
        # Insert learning rate adjustments here using:
        # new_lr = lr_funct(old_lr, i) 
        # K.set_value(train.model_actor.optimizer.lr, new_lr)
        # K.set_value(train.model_critic.optimizer.lr, new_lr)
        # show it with ....
        # print("Setting LR for actor network to",K.get_value(train.model_actor.optimizer.lr))
        
        if args.plot:
            plt.cla()
            x_axis=range(args.start_iter,i+1)

            bax1.clear()
            bax1.grid(True)
            bax1.plot(x_axis,actor_loss,'b')
            bax1.set_ylabel('Actor Loss')

            bax2.clear()
            bax2.grid(True)
            bax2.plot(x_axis,critic_loss,'g')
            bax2.set_ylabel('Critic Loss')
            
            bax3.clear()
            bax3.grid(True)
            bax3.plot(x_axis,prob_mass,'r')
            bax3.set_ylabel('Probability Mass')

            bax4.clear()
            bax4.grid(True)
            bax4.plot(x_axis,bid_metric,'k')
            bax4.set_ylabel('Bid Metric')
            bax4.set_xlabel('Iterations')        
            
            plt.draw()
            plt.pause(0.01)
            
            if i%args.save_every == 0 and i > 0:
                plt.savefig('models/actor_critic_pm_iters_{}.png'.format(i))
            
 
