#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import matplotlib.pyplot as plt
import catch5
import c5utils
import time

debug = False # set to False if total_runs is set to more than 1 
num_runs = 0
total_runs = 100000 # set to 100000 to see averages for many runs 
winning_score = 31
tic = time.process_time()
total_time=0
win_total=[0,0]
total_hands=0

team_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]
team_bid_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]
team_game_avg = [c5utils.RunningAvg(),c5utils.RunningAvg()]

while(num_runs < total_runs):
    
    team_total = [0,0]
    num_hands=0  
    
    while team_total[0] < winning_score and team_total[1] < winning_score:
        dealer_id = (num_runs+num_hands)%4
        if debug:
            print("Dealer is player",dealer_id)
        
        players = [catch5.Player(0),catch5.Player(1),catch5.Player(2),catch5.Player(3)]
        dealer = catch5.Dealer(dealer_id)
        
        dealer.dealPreBid(players)
        
        if (debug):
            for i in range(4):
                hand = " "
                print("Player",i,":")
                for j in range(len(players[i].cards)):
                    hand += c5utils.card2str(players[i].cards[j]) + " "
                print(hand)
        
        top_bid = 0
        top_bidder = -1
        suit = 0
        bids=[]
        for i in range(4):
            current_bid,current_suit = players[(dealer_id+1+i)%4].bid(None,top_bid,dealer_id)
            if current_bid > top_bid:
                top_bid = current_bid
                top_bidder = (dealer_id+1+i)%4
                suit = current_suit
            elif current_bid == top_bid and i == 3: #dealer only needs to tie 
                top_bid = current_bid
                top_bidder = (dealer_id+1+i)%4
                suit = current_suit
            bids.append((current_bid,c5utils.char_suits[current_suit],(dealer_id+1+i)%4))
        
        if debug:
            print("Bids are:",bids)
            print("Top bidder:",top_bidder,"Top suit:",c5utils.char_suits[suit])
        final_hands = dealer.dealPostBid(players,suit)
        
        for i in range(4):
            players[i].cards = final_hands[i]
            
        total_scoop = 0
        total_value = 0
        all_cards = []
        for i in range(4):
            all_cards += players[i].cards
            total_scoop += c5utils.numberSuit(players[i].cards,suit)[0]
            for j in range(len(players[i].cards)):
                total_value += c5utils.cardValue(players[i].cards[j],suit)
            if debug:
                hand = " "
                print("Player",i,":")
                for j in range(len(players[i].cards)):
                    hand += c5utils.card2str(players[i].cards[j]) + " "
                print(hand)
            assert(len(players[i].cards)==6)
        if debug:
            print("Number suited:",total_scoop)
        card_set = set(all_cards)
        assert(len(all_cards) == len(card_set) == 24)
        assert(total_value == 9)
        
        # now let's play the hand out
        trick = []
        char_trick=[]
        team_tricks_char=[[],[]]
        team_tricks = [[],[]]
        trick_winner = top_bidder # top bibber starts the action
        lead_suit = suit          # must lead with the suit they bid 
        for i in range(6):
            for j in range(4):
                curr_card=players[(trick_winner+j)%4].play(None,lead_suit)
                if j == 0:
                    lead_suit = int(curr_card/13)
                trick.append(curr_card)
                char_trick.append(c5utils.card2str(curr_card))   
            trick_winner=(dealer.evalTrick(trick,suit)+trick_winner)%4
            if debug:
                print("Trick",i,":",char_trick)
                print("Trick winner:",trick_winner)
            team_tricks_char[(trick_winner)%2].append(char_trick)
            team_tricks[(trick_winner)%2]+=trick
            trick=[]
            char_trick=[]
            lead_suit=-1
            
        team_val = [0,0]
        team_points = [0,0]
        for i in range(2):
            for c in team_tricks[i]:
                team_val[i] += c5utils.cardValue(c,suit)
        if debug:        
            print("Team0 tricks:",team_tricks_char[0], "Team1 tricks:",team_tricks_char[1])
            print("Team0 trick points:", team_val[0],"Team1 trick points:",team_val[1])
        assert((team_val[0]+team_val[1]) == 9)
        bidding_team = top_bidder%2
        team_points[(bidding_team+1)%2] += team_val[(bidding_team+1)%2]
        team_bid_avg[0].set_avg(team_val[(bidding_team)%2])
        team_bid_avg[1].set_avg(team_val[(bidding_team+1)%2])      
        if team_val[bidding_team] >= top_bid:
            team_points[(bidding_team)%2] += team_val[(bidding_team)%2]
        else:
            team_points[(bidding_team)%2] -= top_bid
        if debug:
            print("Team0 final points:", team_points[0],"Team1 final points:",team_points[1])
        team_avg[0].set_avg(team_points[0])
        team_avg[1].set_avg(team_points[1])
        team_total[0]+=team_points[0]
        team_total[1]+=team_points[1]
        num_hands+=1
        if team_total[0]>=winning_score and team_total[0] > team_total[1]:
            win_total[0]+=1
        if team_total[1]>=winning_score and team_total[1] > team_total[0]:
            win_total[1] +=1
    total_hands += num_hands
    num_runs +=1
    if debug:
        print("Final: Team0:",team_total[0],"Final: Team1:",team_total[1],"Number of hands:",num_hands)
    if (num_runs%1000 == 0):
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