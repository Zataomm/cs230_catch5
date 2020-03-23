#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:19:17 2020

@author: jmonroe
"""
import argparse
import numpy as np
import c5_simulation



# set up simple argument parser 
parser = argparse.ArgumentParser(description='Parser for catch5 play script')
parser.add_argument('-p1', action='store',
                    default='random',type=str,
                    dest='policy1',help='Model file for policy for Team1')
parser.add_argument('-p2', action='store',
                    default='random',type=str,
                    dest='policy2',help='Model file for policy for Team2')
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

    print("policy for Team1:",args.policy1)
    print("policy for Team2:",args.policy2)
    print("Total games to play:",args.total_games)
    print("Debug flag:",args.debug)
    print("Allow random players to bid:",args.random_bid)
    print("Allow random players to select random suits:",args.random_suit)   
    sim=c5_simulation.run_simulations(policy_def={0:args.policy1,1:args.policy2},allow_random_bidding=args.random_bid,
                        allow_random_suit=args.random_suit,DEBUG=args.debug,TOTAL_GAMES=args.total_games,
                        USE_INT_STATES=args.intstate,STATE_DIMS=args.state_dims,ACT_TYPE=args.act_type)
    sim.set_policies()
    stats_dict = sim.play_games()

