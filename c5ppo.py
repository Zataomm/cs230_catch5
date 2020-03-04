import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import backend as K

n_actions=64
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


 
def get_model_actor(input_dims, output_dims):
    state_input = Input(shape=input_dims)
    #available_actions = Input(shape=(1,n_actions))
 
    # Classification block
    x = Dense(512, activation='relu', name='fc1',kernel_initializer='glorot_normal')(state_input)
    x = Dense(256, activation='relu', name='fc2',kernel_initializer='glorot_normal')(x)
    x = Dense(128, activation='relu', name='fc3',kernel_initializer='glorot_normal')(x)    
    #x = Dense(n_actions, activation='linear', name='fc3')(x) 
    #filtered_actions = Add()([x, available_actions])
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input],outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model



def get_model_critic(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(512, activation='relu', name='fc1',kernel_initializer='glorot_normal')(state_input)
    x = Dense(256, activation='relu', name='fc2',kernel_initializer='glorot_normal')(x)
    x = Dense(128, activation='relu', name='fc3',kernel_initializer='glorot_normal')(x)    
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model


def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot


"""
state_dims = (1,42)
n_actions = 64
 
model_actor = get_model_actor(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic(input_dims=state_dims)

state = np.zeros((1,42),dtype='float32')
state_input = K.eval(K.expand_dims(state, 0))
q_value = model_critic.predict([state_input], steps=1)

print(q_value)
"""
