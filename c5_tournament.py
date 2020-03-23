#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import matplotlib.pyplot as plt
import argparse
import numpy as np
import c5_simulation

# set up simple argument parser 
parser = argparse.ArgumentParser(description='Parser for catch5 play script')
parser.add_argument('-p1', action='store',
                    default='random',type=str,
                    dest='policy1',help='Model file for policy for team to be tested')

parser.add_argument('-start', action='store',
                    default=0,type=int,
                    dest='start_iters',help='Iterations to start testing from')

parser.add_argument('-stop', action='store',
                    default=0,type=int,
                    dest='stop_iters',help='Iterations to stop testing from')

parser.add_argument('-step', action='store',
                    default=0,type=int,
                    dest='steps',help='Steps for iterations')

parser.add_argument('-tg', action='store',
                    type=int,
                    default=1,
                    dest='total_games',
                    help='Total games to play')
parser.add_argument('-debug', action='store_true',
                    default=False,
                    dest='debug',
                    help='Turn on debugging')
parser.add_argument('-rb', action='store_false',
                    default=True,
                    dest='random_bid',
                    help='Turn off random bidding')
parser.add_argument('-rs', action='store_false',
                    default=True,
                    dest='random_suit',
                    help='Turn off random suit selection - random player will select suit of highest number in hand.')
parser.add_argument('-pm', action='store_false',
                    default=True,
                    dest='pick_max',
                    help='Turn off selection of max probability and select according to the actual distrubution.')
parser.add_argument('-intstate', action='store_true',
                    default=False,
                    dest='intstate',
                    help='Train with 42 dimensional integer states instead of default 504 dimensional binary states')
parser.add_argument('-state_dims', action='store',
                    type=int,
                    default=504,
                    dest='state_dims',
                    help='Input state dimensions for NN - set to 42 if using -intstate option.')
parser.add_argument('-activation', action='store',
                    type=str,
                    default="tanh",
                    dest='act_type',
                    help='Default is tanh, use \'leaky\' for Leay ReLU activations.')



if __name__ == "__main__":


    args = parser.parse_args()
    all_stats={}
    
    print("Test policy:",args.policy1)
    print("Testing against policies from",args.start_iters,"to ",args.stop_iters,"in increments of ",args.steps)
    print("Total games to play:",args.total_games)
    print("Debug flag:",args.debug)
    print("Allow random players to bid:",args.random_bid)
    print("Allow random players to select random suits:",args.random_suit)   
  
    #Run tests 
    for i in range(args.start_iters,args.stop_iters+1,args.steps):
        current_policy = 'models/policy_{}.hdf5'.format(i)
        print("Testing policy ",i," vs ",args.policy1)
        sim=c5_simulation.run_simulations(policy_def={0:args.policy1,1:current_policy},allow_random_bidding=args.random_bid,
                                          allow_random_suit=args.random_suit,DEBUG=args.debug,TOTAL_GAMES=args.total_games,
                                          USE_INT_STATES=args.intstate,STATE_DIMS=args.state_dims,ACT_TYPE=args.act_type)
        sim.set_policies()
        all_stats[i] = sim.play_games()

    #set x-axis
    x_axis =  list(range(args.start_iters,args.stop_iters+1,args.steps))
    #gather data
    bids_won=[[],[],[],[]]
    average_bid=[[],[],[],[]]
    rewards_per_bid=[[],[],[],[]]
    bid_suit_distribution=[[],[],[],[]]
    hands_won_per_team=[[],[]]
    raw_hands_won_per_team=[[],[]]
    for i in range(args.start_iters,args.stop_iters+1,args.steps):
        for j in range(4):
            bids_won[j].append(all_stats[i]['bids_won'][j])
            average_bid[j].append(all_stats[i]['average_bid'][j])
            rewards_per_bid[j].append(all_stats[i]['rewards_per_bid'][j])
            bid_suit_distribution[j].append(all_stats[i]['bid_suit_distribution'][j])
        for j in range(2):
            hands_won_per_team[j].append(all_stats[i]['hands_won_per_team'][j])
            raw_hands_won_per_team[j].append(all_stats[i]['raw_hands_won_per_team'][j])

    # Now plot the bidding data
    bfig = plt.figure()
    bfig.suptitle('Bidding Information', fontsize=16)
    
    bax1 = bfig.add_subplot(221)
    bax2 = bfig.add_subplot(222)
    bax3 = bfig.add_subplot(223)
    bax4 = bfig.add_subplot(224)
    
    
    bax1.plot(x_axis,bids_won[0],'b')
    bax1.plot(x_axis,bids_won[1],'r')
    bax1.plot(x_axis,bids_won[2],'c')
    bax1.plot(x_axis,bids_won[3],'m')
    bax1.set_ylabel('# Bids Won')
    
    bax2.plot(x_axis,average_bid[0],'b')
    bax2.plot(x_axis,average_bid[1],'r')
    bax2.plot(x_axis,average_bid[2],'c')
    bax2.plot(x_axis,average_bid[3],'m')
    bax2.set_ylabel('# Average Bid')
    
    bax3.plot(x_axis,rewards_per_bid[0],'b')
    bax3.plot(x_axis,rewards_per_bid[1],'r')
    bax3.plot(x_axis,rewards_per_bid[2],'c')
    bax3.plot(x_axis,rewards_per_bid[3],'m')
    bax3.set_ylabel('# Average Rewards Per Bid')
    
    bax4.plot(x_axis,bid_suit_distribution[0],'olive')
    bax4.plot(x_axis,bid_suit_distribution[1],'lime')
    bax4.plot(x_axis,bid_suit_distribution[2],'green')
    bax4.plot(x_axis,bid_suit_distribution[3],'teal')
    bax4.set_ylabel('Bid Suit Distribution')

    plt.show()


    #Now plot the game data 
    
    gfig =  plt.figure()

    gfig.suptitle('Game Information', fontsize=16)
    
    gax1 = gfig.add_subplot(121)
    gax2 = gfig.add_subplot(122)

    gax1.plot(x_axis,hands_won_per_team[0],'b')
    gax1.plot(x_axis,hands_won_per_team[1],'r')
    gax1.set_ylabel('# Hands Won Per Team')

    gax2.plot(x_axis,raw_hands_won_per_team[0],'b')
    gax2.plot(x_axis,raw_hands_won_per_team[1],'r')
    gax2.set_ylabel('# Raw Hands Won Per Team')

    plt.show()
    
    
