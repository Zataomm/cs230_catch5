import numpy as np

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Activation,
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

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
    available_actions = Input(shape=(1,n_actions))
 
    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(n_actions, activation='linear', name='fc3')(x) 
    filtered_actions = keras.layers.Add()([x, available_actions])
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(filtered_actions)

    model = Model(inputs=[state_input, available_actions],outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse'
    model.summary()
    return model



def get_model_critic(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(512, activation='relu', name='fc1')(state_input)
    x = Dense(256, activation='relu', name='fc2')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model


def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 20:
            break
    return total_reward


def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot

# Create environment here
 
state = env.reset()
state_dims = (1,42)
n_actions = 62

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

tensor_board = TensorBoard(log_dir='./logs')

model_actor = get_model_actor_simple(input_dims=state_dims, output_dims=n_actions)
model_critic = get_model_critic_simple(input_dims=state_dims)

ppo_steps = 128
target_reached = False
best_reward = 0
iters = 0
max_iters = 50

while not target_reached and iters < max_iters:

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    for itr in range(ppo_steps):
        state_input = K.expand_dims(state, 0)
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        q_value = model_critic.predict([state_input], steps=1)
        action = np.random.choice(n_actions, p=action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1

        observation, reward, done, info = env.step(action)
        print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))
        mask = not done

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)

        state = observation
        if done:
            env.reset()

    q_value = model_critic.predict(state_input, steps=1)
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    actor_loss = model_actor.fit(
        [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
        [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8,
        callbacks=[tensor_board])
    critic_loss = model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                   verbose=True, callbacks=[tensor_board])

    avg_reward = np.mean([test_reward() for _ in range(5)])
    print('total test reward=' + str(avg_reward))
    if avg_reward > best_reward:
        print('best reward=' + str(avg_reward))
        model_actor.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
        model_critic.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
        best_reward = avg_reward
    if best_reward > 0.9 or iters > max_iters:
        target_reached = True
    iters += 1
    env.reset()

env.close()
