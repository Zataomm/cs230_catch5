#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:13:39 2020

@author: jmonroe


Implementation notes:
    
    Cards are indexed from 0 to 51 by 2,3,4,5,....,K,A and by suit in alphebetical order 

    State is a (1,272) vector containing 5 decks of 52 cards (260), plus 10 elements at the
    end.  The 5 decks represent: 
            Deck 1: Current board
            Deck 2: Players current cards
            Deck 3,4,5: Other players observed discards (in clockwise order from players position)
            Deck 6: All discards observed to this point
    The remaining 12 elements at the end are are 7 for the bid (3 to 9) and 1 for the bidding team (1=your team, 0 = opponent)
    and 1 out of 4 inidcating the scoop suit (top suit in the deck called by the winning bidder) (C,D,H,S)
    
    bidState is a (1,60) vector representing your current hand (52) plus 7 indicating the current bid, plus one indicating if your
    opponent has the current bid or not (1 = your opponent has the bid, 0 = they do not)
    
    board is a dictionary containg the "lead_player" and a list of cards "played"


"""


from random import randint,shuffle 
import numpy as np 
import c5utils

class Player():
    """ Player stores the player's state, cards (for convience),and id 
    
        methods are:
            updateState:  update state given the current board
            play:  play a card given the current board and policy (take a step)  
            bid: make a bid given the current bids and a policy (bid includes number and suit)
    """
    player_id=None

    def __init__(self, player_id):
        """ Every player should have a unique player id
        their own state based on what they have observed 
        """
        self.player_id = player_id
        self.cards = []
        self.state = np.zeros((1,324))
        self.bidState = np.zeros((1,60))
    
    def updateStateBoard(self,board):
        """ Takes in current board and updates players state to 
        reflect the information contained in the hand 
        """
        raise NotImplementedError
        
    def updateStateInitial(self):
        """ Takes initial deal of cards and updates players state to 
        reflect the information contained in their hand 
        """
        raise NotImplementedError
        
    def updateStateBid(self): 
        """ Takes bidding information seen so far to update 
             players state prior to making their bid 
        """   
        raise NotImplementedError    
        
    def play(self, policy, suit_led):
        """ Plays the appropriate card in a players hand according to the
            players policy. We add suit_lead to determine the legal actions. 
        """
        #random play for now for testing
        # check if you have to follow suit 
        num_suit,indx = c5utils.numberSuit(self.cards,suit_led)
        if num_suit > 0:
            card_to_play = self.cards.pop(randint(indx[0],indx[1]))
            #update state 
        else:
            card_to_play = self.cards.pop(randint(0,len(self.cards)-1))
            #update state
        return card_to_play
        
    def bid(self, policy, current_bid, dealer_id):
        """ Plays the appropriate card in a players hand according to the
            current board, using the players policy 
        """
        # Implement random bidding for now to test game logic 
        
        bid = 0
        suit = -1
        if current_bid == 0 and self.player_id == dealer_id:
            bid = 3
        else:
            current_bid = max(2,current_bid)
            # randon bidding for now - first decide to bid
            offer_bid = randint(0,1)
            if offer_bid == 1:
                if self.player_id == dealer_id:
                    bid = current_bid
                elif current_bid < 9:
                    bid = randint(current_bid+1,9)
        if bid > 0: # offer suit as well
            maxCards=0
            for i in range(4):
                currCards,_ = c5utils.numberSuit(self.cards,i)
                if currCards>maxCards:
                    maxCards = currCards
                    suit = i
        
                    
        return bid,suit
        
class Dealer():
    """ Dealer of playing cards """
      
    dealer_id = None 
    
    def __init__(self,dealer_id):
        self.cardsRemaining = 52
        self.deck =list(range(52))
        self.discarded = []
        self.dealer_id = dealer_id

        
    def dealCard(self):
        """ deal a card from the remaining card in the deck.
        Return the card and the card value. """
        if self.cardsRemaining == 0:
            card = -1
        else:
            card = self.deck.pop(randint(0,self.cardsRemaining-1))
            self.cardsRemaining -= 1 
        #print("Num remaining = ",self.cardsRemaining)         
        return card

    def dealPreBid(self,players):
        """ Deal initial 9 cards to players 
        """       
        for j in range(len(players)):
            for i in range(9):
                players[j].cards.append(self.dealCard())
            players[j].cards.sort()

    def dealPostBid(self,players,scoop_suit):
        """ Deal plus up of cards until each player has 6 cards.
        """       
        new_hands = []
        # keep scoop suits and track discarded cards as well 
        for j in range(len(players)):
            num_scoop,indxs = c5utils.numberSuit(players[j].cards,scoop_suit)         
            if num_scoop > 0:
                new_hands.append(players[j].cards[indxs[0]:indxs[1]+1])
                if (indxs[0] > 0):
                    self.discarded +=  players[j].cards[0:indxs[0]]
                if (indxs[1] < len(players[j].cards)):
                    self.discarded +=  players[j].cards[indxs[1]+1:len(players[j].cards)+1]
            else:
                new_hands.append([])
                self.discarded += players[j].cards
                
        # fill in the rest of the players hands depending on who dealt
        for j in range(self.dealer_id+1,self.dealer_id+1+len(new_hands)):
            for i in range(len(new_hands[j%4]),6):
                newCard = self.dealCard()
                if newCard >= 0:
                    new_hands[j%4].append(newCard)
                    
        for j in range(len(new_hands)):
            while len(new_hands[j]) > 6:
                for i in range(len(new_hands[j])):
                    if c5utils.cardValue(new_hands[j][i],scoop_suit) == 0:
                        self.discarded.append(new_hands[j][i])
                        new_hands[j].pop(i)
                        break            
        
        # finish up - check for cards remaining in the deck and make sure everyone has 6 cards 
        if len(self.deck) == 0:
            self.discarded.sort()
            self.deck = self.discarded
            self.cardsRemaining = len(self.deck)
            for j in range(self.dealer_id+1,self.dealer_id+1+len(new_hands)):
                for i in range(len(new_hands[j%4]),6):
                    newCard = self.dealCard()
                    if newCard >= 0:
                        new_hands[j%4].append(newCard)
        # make sure no scoop cards are in the deck or discard pile - if so dealer takes them
        else:
            num_scoop,indxs = c5utils.numberSuit(self.deck,scoop_suit)
            if num_scoop > 0:
                #print("Init-dealer hand:",new_hands[self.dealer_id])
                extra_scoop = self.deck[indxs[0]:indxs[1]+1]
                #print("Extra in deck",extra_scoop)
                total_value = 0
                for c in extra_scoop:
                    total_value += c5utils.cardValue(c,scoop_suit)
                #print("There were ",total_value," points left in deck.")   
                # dealer will take these cards 
                new_hands[self.dealer_id] += extra_scoop
                new_hands[self.dealer_id].sort()
                #print("Dealer hand + scoop:",new_hands[self.dealer_id])
                dealer_num_scoop,dindxs = c5utils.numberSuit(new_hands[self.dealer_id],scoop_suit)
                dealer_scoop = new_hands[self.dealer_id][dindxs[0]:dindxs[1]+1]
                #print("Dealer scoop:",dealer_scoop)
                if dealer_num_scoop > 6:
                    while len(dealer_scoop) > 6:
                        for i in range(len(dealer_scoop)):
                            if c5utils.cardValue(dealer_scoop[i],scoop_suit) == 0:
                                dealer_scoop.pop(i)
                                break
                    new_hands[self.dealer_id] = dealer_scoop                                       
                else:
                    dealer_non_scoop = []
                    #print("indicies:",dindxs, "length:",len(new_hands[self.dealer_id]))
                    if dindxs[0] > 0:
                        dealer_non_scoop +=  new_hands[self.dealer_id][0:dindxs[0]]
                    if dindxs[1] < len(new_hands[self.dealer_id]):
                        dealer_non_scoop +=  new_hands[self.dealer_id][dindxs[1]+1:len(new_hands[self.dealer_id])+1]
                    shuffle(dealer_non_scoop)
                    #print("Dealer non_scoop:",dealer_non_scoop)
                    new_hands[self.dealer_id] = dealer_scoop+dealer_non_scoop
                    new_hands[self.dealer_id] = new_hands[self.dealer_id][0:6]
                #print("Dealer final hand",new_hands[self.dealer_id])
        for j in range(len(new_hands)):
            new_hands[j].sort()
            
        return new_hands  
    
    def evalCards(self,card0,card1,scoop_suit):
        """ Determine best card and return 0 if card0 is best 1, if card 1,
            taking into consideration the scoop suit 
        """
        suit0 = int(card0/13)
        suit1 = int(card1/13)
        if suit0 == suit1:
            return card1 > card0
        elif suit0 == scoop_suit:
            return 0
        elif suit1 == scoop_suit:
            return 1
        else:
            return 0 
        
    
    def evalTrick(self,trick,scoop_suit):
        """ Takes in a trick in the order played and returns the player that 
            won the trick based in the suit led and the scoop suit 
        """
        top_card, top_player = trick[0],0
        for i in range(1,4):
            if self.evalCards(top_card,trick[i],scoop_suit) == 1:
                top_card,top_player = trick[i],i
        return top_player
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


