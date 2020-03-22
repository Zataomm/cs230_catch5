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


class random_play():
    """ Implements random play for players in tournament."""

    def __init__(self,random_bidding=True,random_suit=True):

        self.random_bidding=random_bidding
        self.random_suit=random_suit
        
    def play(self,state_input,int_state,legal_actions):
        action=c5utils.random_action(legal_actions)
        if not self.random_bidding:
            if action <8:
                if legal_actions[0] > 0: #pass if possible
                    action=0
                else:
                    action=1  # min-bid
        if not self.random_suit:
            if action < 12 and action >=8: #suit selection
                cards = int_state[9:18]
                high_suit=-1
                num_in_suit=-1
                for i in range(4):
                     ns,_= c5utils.numberSuit(cards,i)
                     if ns > num_in_suit:
                         num_in_suit=ns
                         high_suit=i
                action = 8+high_suit
        return action


class policy_play():
    """ Implements play via policy for players in tournament."""

    def __init__(self,env,policy,nactions,pick_max=True):
        
        self.pick_max=pick_max
        self.policy = policy
        self.nactions=nactions
        self.env = env
        
    def play(self,state_input,int_state,legal_actions):
        action_dist = self.policy.predict([state_input], steps=1)
        legal_action_dist,_,_,_=self.env.adjust_probs(np.squeeze(action_dist,axis=0),legal_actions)
        if self.pick_max:
            action = np.argmax(legal_action_dist[ :])
        else:
            action = np.random.choice(N_ACTIONS, p=legal_action_dist[ :])
        return action 
            
class run_simulations():
    """ Class used to run simulations for catch5 - using a random player, as well as being able to 
        load players policy's from input files, etc....  Program will keep meaningful stats in order 
        to be able to tell if networks are improving.  
    """
    def __init__(self,DEBUG=True,TOTAL_GAMES=10,policy_def={0:"random",1:"random"},allow_random_bidding=True,
                 allow_random_suit=True,STATE_DIMS=504,USE_INT_STATES=False,ACT_TYPE="tanh",PICK_MAX=True):

        # parameters
        self.DEBUG = DEBUG
        self.TOTAL_GAMES=TOTAL_GAMES
        self.policy_def=policy_def
        self.player_policy=[None,None]
        self.nn_policy=[None,None]
        self.STATE_DIMS = STATE_DIMS
        self.N_ACTIONS = 64
        self.winning_score=31
        self.allow_random_bidding=allow_random_bidding
        self.allow_random_suit=allow_random_suit
        self.dummy_val=1.0
        self.env=catch5_env.catch5()
        self.USE_INT_STATES=USE_INT_STATES
        self.act_type=ACT_TYPE
        self.pick_max=PICK_MAX

        
        #stats and counters
        self.score=[0,0]
        self.total_hands=0
        self.game_win_total=[0,0]
        self.hand_win_total=[0,0]
        self.raw_hand_win_total=[0,0]
        self.points_per_game=[c5utils.RunningAvg(),c5utils.RunningAvg()]
        self.hands_per_game=[c5utils.RunningAvg()]
        self.number_of_bids_won = [0,0,0,0]
        self.average_winning_bid = [c5utils.RunningAvg(),c5utils.RunningAvg(),c5utils.RunningAvg(),c5utils.RunningAvg()]
        self.average_rewards_per_winning_bid=[c5utils.RunningAvg(),c5utils.RunningAvg(),
                                              c5utils.RunningAvg(),c5utils.RunningAvg()]
        

    def set_policies(self):
        # loop through the list and set policies for each of the players
        for i in range(2):
            if self.policy_def[i] == "random":
                print("Setting team:",i,"to random.")
                self.player_policy[i]=random_play(random_bidding=self.allow_random_bidding,random_suit=self.allow_random_suit)
            else: # policy is defined by network weights
                print("Loading weights from:",self.policy_def[i],"into network for player",i)
                _,self.nn_policy[i]=c5ppo.build_actor_network(input_dims=self.STATE_DIMS,output_dims=self.N_ACTIONS,
                                                              learning_rate=self.dummy_val,clipping_val=self.dummy_val,entropy_beta=self.dummy_val,act_type=self.act_type)
                self.nn_policy[i].load_weights(self.policy_def[i])
                self.player_policy[i]=policy_play(env=self.env,policy=self.nn_policy[i],nactions=self.N_ACTIONS,pick_max=self.pick_max)

    def play_games(self):
        game_num=0
        bid_suits=[0,0,0,0]
        while(game_num < self.TOTAL_GAMES):
            self.score = [0,0]
            self.num_hands=0  

            # play a game 
            while self.score[0] < self.winning_score and self.score[1] < self.winning_score:

                if self.DEBUG:
                    print("Dealer is player ",self.env.dealer)
                #play a hand 
                done = False

                while not done:
                    observation = self.env.states[self.env.current_player]
                    int_obs =  np.copy(self.env.int_states[self.env.current_player])
                    if not self.USE_INT_STATES:
                        state_input = observation[np.newaxis,:]
                    else:
                        state_input = int_obs[np.newaxis,:]

                    if self.DEBUG:
                        c5utils.print_binstate(observation,self.env.current_player)

                    legal_actions=self.env.legal_actions()
        
                    if self.DEBUG:
                        c5utils.print_actions(legal_actions)

                    # now get the next move - and update states depending on who is playing
                    action=self.player_policy[self.env.current_player%2].play(state_input,int_obs,legal_actions)

                                    
                    if self.DEBUG:
                        c5utils.print_action(action)
                        print("Step number = ",self.env.num_plays)

                    #take a step and return next state 
                    observation,reward,done,_ = self.env.step(self.env.action_map[action])


                self.num_hands+=1

                self.number_of_bids_won[self.env.bidder]+=1
                self.average_winning_bid[self.env.bidder].set_avg(self.env.best_bid)
                self.average_rewards_per_winning_bid[self.env.bidder].set_avg(self.env.rewards[self.env.bidder])
                bid_suits[int(self.env.bid_suit)]+=1
                if self.DEBUG:
                    c5utils.print_tricks(self.env.trick_info)

                if self.DEBUG:
                    for i in range(4):
                        print("Rewards for team ",i,":",self.env.rewards[i])

                self.score[0]+=self.env.rewards[0]
                self.score[1]+=self.env.rewards[1]

                if self.env.rewards[0] > self.env.rewards[1]:
                    self.hand_win_total[0] += 1
                else:
                    self.hand_win_total[1] += 1

                if self.env.game_points[0] > self.env.game_points[1]:
                    self.raw_hand_win_total[0] += 1
                else:
                    self.raw_hand_win_total[1] += 1

                    

                if self.DEBUG:
                    print("====================  Scores team[0]:",self.score[0])
                    print("====================  Scores team[1]:",self.score[1])


                #reset and count hands 
                self.env.reset()        

                if self.score[0]>=self.winning_score and self.score[0] > self.score[1]:
                    self.game_win_total[0]+=1
                if self.score[1]>=self.winning_score and self.score[1] > self.score[0]:
                    self.game_win_total[1] +=1

            self.hands_per_game[0].set_avg(self.num_hands)
            self.points_per_game[0].set_avg(self.score[0])
            self.points_per_game[1].set_avg(self.score[1])

            game_num+=1


            print("Current score: Team 0:",self.game_win_total[0],"Team 1:",self.game_win_total[1])

        for i in range(4):
            print("Player:",i,"number of winning bids:",self.number_of_bids_won[i])
            print("Average winning bid:",self.average_winning_bid[i].get_avg())
            print("Avg rewards per winning bid:",self.average_rewards_per_winning_bid[i].get_avg())
            
        print("Bid suit distribution:",bid_suits)

        print("Wins for team 0:",self.game_win_total[0])
        print("Wins for team 1:",self.game_win_total[1])
        print("Hands won for team 0:",self.hand_win_total[0])
        print("Hands won for team 1:",self.hand_win_total[1])
        print("Raw hands won for team 0:",self.raw_hand_win_total[0])
        print("Raw hands won for team 1:",self.raw_hand_win_total[1])        
        print("Point avg Team0:",self.points_per_game[0].get_avg())
        print("Point avg Team1:",self.points_per_game[1].get_avg()) 
        print("Average number of hands per game:",self.hands_per_game[0].get_avg())
            
