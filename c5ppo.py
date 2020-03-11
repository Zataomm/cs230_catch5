import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
import sys

n_actions=64
clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.01
gamma = 0.99
lmbda = 0.95
learning_rate = 0.00005

#turn off warnings and above 
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#turn off eager execution 
tf.compat.v1.disable_eager_execution()


def get_simple_advantages(values, rewards):
    """ Classical GP advantages """
    returns = []
    adv=[]
    r = rewards[-1]
    T=len(rewards)
    for i in range(T):
        returns.append((gamma)**(T-1-i)*r)
    for i in range(T):
        adv.append(returns[i] - values[i])    
    return returns,adv
        
def get_advantages(values, rewards):
    """ We can cheat on calculating the advatages since we have a fixed episode length every time
    """
    returns = []
    adv=[]
    gae = 0
    values = values+[0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lmbda * gae
        adv.insert(0,gae)
        returns.insert(0, gae + values[i])
    return returns, adv


def ppo_loss(oldpolicy_probs, advantages, returns, values):
    def loss(y_true, y_pred):
        newpolicy_probs = K.sum(y_true * y_pred,axis=1)
        old_probs = K.sum(y_true * oldpolicy_probs,axis=1)
        
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(old_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(returns - values))
        entropy_loss =  K.mean(-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta *entropy_loss
        return total_loss
    return loss



def build_actor_critic_network(input_dims,output_dims):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims))
    advantages = Input(shape=(1,1))
    returns = Input(shape=(1,1))
    values = Input(shape=(1,1))

    # Classification block
    dense1 = Dense(512, activation='relu', name='fc1',kernel_initializer='glorot_normal')(state_input)
    dense2 = Dense(512, activation='relu', name='fc2',kernel_initializer='glorot_normal')(dense1)
    dense3 = Dense(256, activation='relu', name='fc3',kernel_initializer='glorot_normal')(dense2)
    dense4 = Dense(256, activation='relu', name='fc4',kernel_initializer='glorot_normal')(dense3)  
    pred_probs = Dense(output_dims, activation='softmax', name='actor_predictions')(dense4)
    pred_value = Dense(1, activation='tanh',name='critic_values')(dense4)

    
    actor = Model(inputs=[state_input,oldpolicy_probs,advantages,returns,values],outputs=[pred_probs])
    actor.compile(optimizer=Adam(lr=learning_rate), loss=[ppo_loss(oldpolicy_probs=oldpolicy_probs,
                                                         advantages=advantages,returns=returns,values=values)])
    actor.summary()
    
    critic = Model(inputs=[state_input], outputs=[pred_value])
    critic.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    critic.summary()
    
    policy = Model(inputs=[state_input],outputs=[pred_probs])
    
    return actor,critic,policy

