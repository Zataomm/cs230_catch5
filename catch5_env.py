#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:42:35 2020

@author: eric

"""
import numpy as np 
import c5utils


class catch5():
    """ General environment to play the catch 5 game - to include 
        keeping state and implementing the reset and step functions for 
        playing the game and generating trajectories. Current representation
        of input and output states for the Actor NN:
        
        Int State: [BIDS(4),BID_SUIT(1),CARDS_IN_TRICK(4),CARDS_IN_HAND(9),
                CARDS_DISCARDED(6), CARDS_OPPONENTS_DISCARDED(6)x3] Dim=42
        Bin State: [BIDS(4*8),BID_SUIT(1*4),CARDS_IN_TRICK(4*52),CARDS_IN_HAND(52),
                CARDS_DISCARDED(52), CARDS_OPPONENTS_DISCARDED(52)x3] Dim=504
        Action: [BIDS:PASS,3,4,5,6,7,8,9
                PICK_SUIT: C,D,H,S
                Play Card: 2C,3C,...,AC,2D,3D,....,AD,2H,3H,...,AH,2S,...,AS]
                Dim=64   
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """ Reset environment for the start of a round. 
            Consists of reshuffeling deck - dealing initial 
            hands to players, and restting thier states to 
            contain only their initial 9 cards."""
        self.int_input_dim=42
        self.input_dim=504
        self.softmax_dim=64
        self.action_map=[1,3,4,5,6,7,8,9,1,2,3,4]+list(range(1,53))
        #reset deck 
        self.shuffled_deck = list(np.random.permutation(52) + 1)
        #reset players states to zero 
        self.int_states = [np.zeros((1,self.int_input_dim)),np.zeros((1,self.int_input_dim)),
                          np.zeros((1,self.int_input_dim)),np.zeros((1,self.int_input_dim))]
        self.states = [np.zeros((1,self.input_dim)),np.zeros((1,self.input_dim)),
                          np.zeros((1,self.input_dim)),np.zeros((1,self.input_dim))]        
        # Choose random player to start the game ...
        self.dealer = np.random.randint(4)
        self.current_player = (self.dealer+1)%4
        #Deal initial cards to all players to set initial states 
        for i in range(4):
            self.int_states[i][0,9:18]=np.sort(self.shuffled_deck[i*9:9*(i+1)])
            for c in self.shuffled_deck[i*9:9*(i+1)]:
                self.states[i][0,36+4*52+c-1]=1
        self.shuffled_deck = self.shuffled_deck[36:]
        self.cards_remaining = 16
        self.num_plays=0
        self.trick_info=[]
        self.bidder=-1
        self.best_bid=-1
        self.rewards=[0,0,0,0]
        self.game_points=[0,0,0,0]
        self.penalty=[0,0,0,0]
        self.a2i={1:0,3:1,4:2,5:3,6:4,7:5,8:6,9:7}
        
        return self.current_player

    def adjust_probs(self,probs,legal_moves):
        """ Take in softmax output and adjust based on legal moves.
        """
        new_probs = np.multiply(probs,legal_moves)
        norm = np.sum(new_probs)
        if (norm>0):
            new_probs = new_probs/norm
        else: # choose random legal action
            num_moves = np.sum(legal_moves)
            new_probs = legal_moves/num_moves
        return new_probs
    
    def legal_actions(self):
        """ Determine legal actions for the given input state.
            This is done by looking at different values of the current state 
            to determine the progress of the game (bidding, tricks, lead card, etc)
        """
        actions = np.full((1,self.softmax_dim),0)
        ones = np.ones((1,8))
        if self.int_states[self.current_player][0,0]==0:  # still in bidding phase - first 8 are for bidding
            actions[0,0:8] = ones[0,0:8]
            bid_sum=0
            for i in range(1,4):
                bid_sum += self.int_states[(self.current_player+i)%4][0,0]
            if bid_sum ==3: # all players passed - then dealer must bid - pass is not allowed
                actions[0,0]=0
        elif self.int_states[self.current_player][0,4] == 0:  #winning bidder is choosing a suit
            actions[0,8:12] = ones[0,0:4]
        else:  # we are playing so actions have to follow the "follow lead suit rule"
            lead_suit = -1 #find lead suit
            for i in range(3):
                if self.int_states[self.current_player][0,6+i] > 0:
                    lead_suit = int((self.int_states[self.current_player][0,6+i]-1)/13)
                    break
            cards = list(self.int_states[self.current_player][0,11+int(self.num_plays/4):18])
            num,indx = c5utils.numberSuit(cards,lead_suit)
            if num == 0: #any card player holds is legal
                suit_cards=cards
            else:  # we have a card from lead suit and we have to follow
                suit_cards = cards[indx[0]:indx[1]+1]
            for c in suit_cards:
                actions[0,12+int(c)-1]=ones[0,0]               
        return actions
    
    def update_states(self,action):
        """ Update states based on current round and player acting.
            States of all players should be adjusted - even if they 
            have already played - so their states are set for future 
            rounds.  
        """
        if self.int_states[self.current_player][0,0]==0: # action is a bid 
            for i in range(4):
                self.int_states[(self.current_player+i)%4][0,(4-i)%4]=action
                self.states[(self.current_player+i)%4][0,8*((4-i)%4)+self.a2i[action]]=1
        elif self.int_states[self.current_player][0,4] == 0: # action is selecting a suit 
            players_cards=[]                       
            for i in range(4):
                #set action
                self.int_states[(self.current_player+i)%4][0,4]=action
                self.states[(self.current_player+i)%4][0,32+action-1]=1                
                #get cards for a redeal
                players_cards.append(list(self.int_states[(self.dealer+i)%4][0,9:18]))
            # deal new cards based on selected suits 
            new_hands=c5utils.dealPostBid(players_cards,self.shuffled_deck,action-1)
            zeros52=np.zeros((1,52))
            for i in range(4):
                self.int_states[(self.dealer+i)%4][0,9:18]=new_hands[i]
                self.states[(self.dealer+i)%4][0,32+4+4*52:32+4+4*52+52]=zeros52
                for c in new_hands[i]:
                    if c > 0:
                        self.states[(self.dealer+i)%4][0,32+4+4*52+int(c)-1]=1
        else: # action is playing a card - set by round
            # update discards for all the players states
            # current player - remove card from hand and place in discards
            cards=list(self.int_states[self.current_player][0,9:18])
            cards.remove(action)
            cards.append(0)
            cards.sort()
            self.int_states[self.current_player][0,9:18]=cards
            self.states[self.current_player][0,32+4+4*52+action-1]=0
            # add card to everyones discard list - and to trick list 
            for i in range(4):
                j=(4-i)%4
                discards=list(self.int_states[(self.current_player+i)%4][0,18+6*j:24+6*j])
                discards[0]=action
                discards.sort()
                self.int_states[(self.current_player+i)%4][0,18+6*j:24+6*j]=discards
                self.states[(self.current_player+i)%4][0,32+4+52*4+52+52*j+action-1]=1
                #current trick list 
                self.int_states[(self.current_player+i)%4][0,5+j]=action
                self.states[(self.current_player+i)%4][0,32+4+52*j+action-1]=1
                
    def setup_next_round(self):
        """ Check if we are at the end of a round - record winning tricks and
            set up bid winner if done bidding, and set next player in round.
        """
        if (self.num_plays%4) != 0: #not at end of round - keep going
            self.current_player = (self.current_player+1)%4
        elif self.num_plays > 4: # past bidding and at end of hand 
            four_zeros=np.zeros((1,4))
            fourx52_zeros=np.zeros((1,4*52))
            trick = list(self.int_states[self.current_player][0,5:9])
            scoop_suit=self.int_states[self.current_player][0,4]-1
            trick_winner=c5utils.evalTrick(trick,scoop_suit)
            trick_winner=(self.current_player+trick_winner)%4
            self.trick_info.append([trick_winner,trick])
            self.current_player=trick_winner
            for i in range(4):
                self.int_states[i][0,5:9]=four_zeros
                self.states[i][0,(32+4):(32+4+4*52)]=fourx52_zeros
        elif self.num_plays == 4 and self.int_states[self.current_player][0,4] == 0:
            bidder,self.best_bid = c5utils.winning_bidder(list(self.int_states[self.current_player][0,0:4]))
            self.bidder = (self.current_player+bidder)%4
            self.current_player=self.bidder      
            self.num_plays-=1 #keeps the round count correct
                
    def eval_rewards(self):
        """ At end of game - so we need to compute the value of the 
            tricks won in each hand and allocate points accordingly - to 
            include penalty if the teams bid was not reached. We will also 
            set rewards to rewards/9 so that the rewards will be between 
            -1 and +1 and be compatible with the output of tanh function.
        """
        assert(len(self.trick_info)==6)
        scoop_suit=self.int_states[self.current_player][0,4]-1
        for t_inf in self.trick_info:
            trick_val=0
            for cd in t_inf[1]:
                trick_val += c5utils.cardValue(int(cd),scoop_suit)
            self.game_points[t_inf[0]]+=trick_val
        assert((self.game_points[0]+self.game_points[1]+self.game_points[2]+self.game_points[3])==9)
        self.game_points[0]=self.game_points[2]=(self.game_points[0]+self.game_points[2])
        self.game_points[1]=self.game_points[3]=(self.game_points[1]+self.game_points[3])
        if self.best_bid > self.game_points[self.bidder]:
            self.penalty[self.bidder] = self.penalty[(self.bidder+2)%4] =  -self.best_bid
        for i in range(4):
            if self.penalty[i]<0:
                self.rewards[i]=self.penalty[i]
            else:
                self.rewards[i]=self.game_points[i]
        
        
    def step(self,action):
        """ Takes a step in the environment given the action.
            Step consists of updating the states of the players.
            If the step is at the end of a round, then we need to
            evaluate the winner of the past trick and set the player 
            to play next and the next state equal to this player state."""
        # update the states based on this action  
        info = {}
        self.update_states(action)
        self.num_plays +=1
        self.setup_next_round()
        if self.num_plays < 28: # not done 
            done = False
            reward = [0,0,0,0]
            observation = self.states[self.current_player]
        else:
            done=True
            reward=self.eval_rewards()
            observation=-1
        return observation,reward,done,info
    


        
        

