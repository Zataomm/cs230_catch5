#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:13:45 2020

@author: jmonroe
"""
import numpy as np
from random import shuffle 
from bisect import bisect_left,bisect_right

char_suits = ['C','D','H','S']

ppCards = ["2C","3C","4C","5C","6C","7C","8C","9C","TC","JC","QC","KC","AC",
           "2D","3D","4D","5D","6D","7D","8D","9D","TD","JD","QD","KD","AD",
           "2H","3H","4H","5H","6H","7H","8H","9H","TH","JH","QH","KH","AH",
           "2S","3S","4S","5S","6S","7S","8S","9S","TS","JS","QS","KS","AS"]

action_map=[1,3,4,5,6,7,8,9,1,2,3,4]+list(range(1,53))


class RunningAvg():
    """  Keeps track of  running average for computing stats. 
    """
    
    def __init__(self):
        """ Initialize avg to zero and number to zero
        """
        self.avg = 0.0
        self.num = 0
 
    
    def get_avg(self):
        return self.avg
    
    def get_num(self):
        return self.num

    def set_avg(self,new_val):
        self.avg = (float(self.num)*self.avg + new_val)/float(self.num+1)
        self.num += 1
        return self.avg
    


# helper functions 
def card2str(cards):
    """ print cards for ease of debugging 
    """
    cardStr = None
    if type(cards) == int:
        cardStr = ppCards[cards-1]
    else:
        cardStr = None
        for i in len(cards):
            cardStr = cardStr + ppCards[i-1] + ","
    return cardStr

def cardValue(card,suit):
    """ return the value of the card if it is a scoring card in catch5
        scoring cards are: 2,A,10,J = 1pt, and 5 = 5pts
    """
    score = 0
    
    if (card-13*suit) in [1,9,10,13]:
        score = 1
    elif (card-13*suit) == 4:
        score = 5
    
    return score
        

def numberSuit(cards,suit):
    """ determine the number of cards in the list cards that have a given suit
        and return the start and stop index for the cards in the list - assuming 
        cards are numbered from 1 to 52 
    """
    minBound = 13*suit;
    maxBound = minBound+14;   
    
    numSuited = 0
    indxs = (-1,-1)
    
    if suit >= 0 and suit <= 3:
        lowIndx = bisect_right(cards,minBound)
        highIndx = bisect_left(cards,maxBound)
        
        numSuited = highIndx-lowIndx
        
        if numSuited > 0:
            indxs = (lowIndx,highIndx-1)
        else:
            numSuited = 0
    
    return numSuited,indxs

def winning_bidder(bids):
    """ Determine best bid and return player number for this bid
    """
    bidder=0
    best_bid=bids[0]
    for i in range(1,4):
        if bids[i]>best_bid:
            best_bid = bids[i]
            bidder=i
    return bidder,best_bid

def evalCards(card0,card1,scoop_suit):
    """ Determine best card and return 0 if card0 is best 1, if card 1,
        taking into consideration the scoop suit 
    """
    suit0 = int((card0-1)/13)
    suit1 = int((card1-1)/13)
    if suit0 == suit1:
        return card1 > card0
    elif suit0 == scoop_suit:
        return 0
    elif suit1 == scoop_suit:
        return 1
    else:
        return 0 
        
    
def evalTrick(trick,scoop_suit):
    """ Takes in a trick in the order played and returns the player that 
        won the trick based in the suit led and the scoop suit 
    """
    top_card, top_player = trick[0],0
    for i in range(1,4):
        if evalCards(top_card,trick[i],scoop_suit) == 1:
            top_card,top_player = trick[i],i
    return top_player
    

def dealPostBid(players,deck,scoop_suit):
    """ Deal plus up of cards until each player has 6 cards. Dealers 
        hand is epected to be last list in the list of lists in players
    """       
    new_hands = []
    discarded = []
    # keep scoop suits and track discarded cards as well 
    for j in range(len(players)):
        num_scoop,indxs = numberSuit(players[j],scoop_suit)         
        if num_scoop > 0:
            new_hands.append(players[j][indxs[0]:indxs[1]+1])
            if (indxs[0] > 0):
                discarded +=  players[j][0:indxs[0]]
            if (indxs[1] < len(players[j])):
                discarded +=  players[j][indxs[1]+1:len(players[j])+1]
        else:
            new_hands.append([])
            discarded += players[j]

    # get rid of players scoop suit cards that got over 6 in the first deal
    for j in range(len(new_hands)):
        while len(new_hands[j]) > 6:
            for i in range(len(new_hands[j])):
                if cardValue(new_hands[j][i],scoop_suit) == 0:
                    deck.append(new_hands[j][i])
                    new_hands[j].pop(i)
                    break            
   
    shuffle(deck)    
        
    # fill in the rest of the players hands depending on who dealt
    for j in range(len(players)):
        for i in range(len(new_hands[j]),6):
            if len(deck)>0:
                newCard = deck.pop()
                new_hands[j%4].append(newCard)
    
    # finish up - check for cards remaining in the deck and make sure everyone has 6 cards 
    if len(deck) == 0:
        discarded.sort()
        deck = discarded
        for j in range(4):
            for i in range(len(new_hands[j%4]),6):
                if len(deck)>0:
                    newCard = deck.pop()
                if newCard >= 0:
                    new_hands[j%4].append(newCard)
                    
    # make sure no scoop cards are in the deck or discard pile - 
    # if so dealer takes them - to ensure 9 possible points per hand
    else:
        deck.sort()
        num_scoop,indxs = numberSuit(deck,scoop_suit)
        if num_scoop > 0:
            #print("Init-dealer hand:",new_hands[self.dealer_id])
            extra_scoop = deck[indxs[0]:indxs[1]+1]
            #print("Extra in deck",extra_scoop)
            total_value = 0
            for c in extra_scoop:
                total_value += cardValue(c,scoop_suit)
            #print("There were ",total_value," points left in deck.")   
            # dealer will take these cards 
            new_hands[3] += extra_scoop
            new_hands[3].sort()
            #print("Dealer hand + scoop:",new_hands[self.dealer_id])
            dealer_num_scoop,dindxs = numberSuit(new_hands[3],scoop_suit)
            dealer_scoop = new_hands[3][dindxs[0]:dindxs[1]+1]
            #print("Dealer scoop:",dealer_scoop)
            if dealer_num_scoop > 6:
                while len(dealer_scoop) > 6:
                    for i in range(len(dealer_scoop)):
                        if cardValue(dealer_scoop[i],scoop_suit) == 0:
                            dealer_scoop.pop(i)
                            break
                new_hands[3] = dealer_scoop                                       
            else:
                dealer_non_scoop = []
                #print("indicies:",dindxs, "length:",len(new_hands[self.dealer_id]))
                if dindxs[0] > 0:
                    dealer_non_scoop +=  new_hands[3][0:dindxs[0]]
                if dindxs[1] < len(new_hands[3]):
                    dealer_non_scoop +=  new_hands[3][dindxs[1]+1:len(new_hands[3])+1]
                shuffle(dealer_non_scoop)
                #print("Dealer non_scoop:",dealer_non_scoop)
                new_hands[3] = dealer_scoop+dealer_non_scoop
                new_hands[3] = new_hands[3][0:6]
            #print("Dealer final hand",new_hands[self.dealer_id])
    for j in range(len(new_hands)):
        new_hands[j].sort()
        new_hands[j] = [0,0,0]+new_hands[j]
        
    return new_hands  

def random_action(actions):
    legal_acts = np.argwhere(actions==0)
    assert(legal_acts.shape[0]>0)
    rand_action=np.random.randint(legal_acts.shape[0])
    return legal_acts[rand_action][1]

def print_actions(actions):
    bid_str=""
    print("======== Available Actions: ==========")
    for i in range(8):
        if actions[0,i]==1:
            bid_str += str(action_map[i])+","
    print("Bids:",bid_str)
    suit_str=""
    for i in range(8,12):
        if actions[0,i]==1:
            suit_str += char_suits[action_map[i]-1]+","
    print("Suits:",suit_str)
    card_str=""
    for i in range(12,64):
        if actions[0,i]==1:
            card_str += ppCards[action_map[i]-1]+","
    print("Cards:",card_str)
    
def print_action(action):
    print("======== Action Taken: ==========")
    if action < 8:
        print("Bid of",action_map[action])
    elif action < 12:
        print("Scoop suit is",char_suits[action_map[action]-1])
    else:
        print("Play:",ppCards[action_map[action]-1])

def print_state(state,player):
    print("======== State for player:",player," ==========")
    if state[0,0] > 0:
        print("Bid is:",state[0,0])
    else:
        print("Waiting to bid")
    if state[0,4]>0:
        print("Bid suit is:",char_suits[int(state[0,4])-1])
    trk_str=""
    for i in range(5,9):
        if state[0,i]==0:
            trk_str+="X "
        else:
            trk_str+=ppCards[int(state[0,i])-1]+" "        
    print("Cards on table:",trk_str)
    hand=""
    for i in range(9,18):
        if state[0,i]>0:
            hand+=ppCards[int(state[0,i])-1]+" "      
    print("Cards in hand:",hand)
    hand=""
    for i in range(18,24):
        if state[0,i]>0:
            hand+=ppCards[int(state[0,i])-1]+" "      
    print("My discards:",hand)   
    hand=""
    for i in range(24,30):
        if state[0,i]>0:
            hand+=ppCards[int(state[0,i])-1]+" "      
    print("Player",(player+1)%4," discards:",hand)    
    hand=""
    for i in range(30,36):
        if state[0,i]>0:
            hand+=ppCards[int(state[0,i])-1]+" "      
    print("Player",(player+2)%4," discards:",hand)        
    hand=""
    for i in range(36,42):
        if state[0,i]>0:
            hand+=ppCards[int(state[0,i])-1]+" "      
    print("Player",(player+3)%4," discards:",hand)        
    
    
def print_tricks(trick_info):    
    for trick in trick_info:
        trk_str=""
        for c in trick[1]:
            trk_str+=ppCards[int(c)-1]+" "
        print("Player:",trick[0],"trick:",trk_str)
        
    
    

