import pickle
import random

import numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from collections import deque

tf.random.set_seed(0)

env = gym.make('LunarLander-v2')
env.reset()

state_size = env.observation_space.shape
num_actions = env.action_space.n


q_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=state_size),
    tf.keras.layers.Dense(units=124, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=num_actions, activation='linear'),
])

target_q_network = tf.keras.Sequential([
    tf.keras.layers.Input(shape=state_size),
    tf.keras.layers.Dense(units=124, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer="he_normal"),
    tf.keras.layers.Dense(units=num_actions, activation='linear'),
])

target_q_network.set_weights(q_network.get_weights())


#Needs to be in a seperate method to use tf.function (optimizing learning speed)
@tf.function
def training_step(q_target, states, actions, OPTIMIZER, LOSS):
    # Calculate the loss
    with tf.GradientTape() as tape:
        q_values = q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
        # Compute the loss
        loss = LOSS(q_target, q_values)

    # Gradient w.r.t loss function and our models variables
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Gradient descent step - backpropergation
    OPTIMIZER.apply_gradients(zip(gradients, q_network.trainable_variables))


# plotting data from training
# -----------------------------------------------PROGRAM-----------------------------------------------------------------

# MATPLOT (METRICS)
plt.figure(figsize=(8, 6))

# CONSTANTS
BUFFER_SIZE = 100_000  # size of memory buffer
DISCOUNT = 0.995  # locked
ALPHA = 1e-4 # locked
STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

BATCH_SIZE = 64  # mini-batch size
TAU = 1e-3  # soft update parameter
EPSILON_DECAY = 0.990  # ε decay rate for ε-greedy policy
EPSILON_MIN = 0.01  # minimum ε value for ε-greedy policy

MAX_EPISODES = 2000
MAX_STEPS = 1000

# DYNAMIC
experience_buffer = deque(maxlen=BUFFER_SIZE)
epsilon = 1.0

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=ALPHA) # weight_decay=1e-3
LOSS = tf.keras.losses.MSE

num_p_av = 100  # number of total points to use for averaging (might delete)

for i in range(MAX_EPISODES):

        # Reset environment
    state, _ = env.reset()

    for t in range(MAX_STEPS):

            # reshape our
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network -- rename it

            #remove function add it directly
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            Q_values = q_network(state_qn)
            action = np.argmax(Q_values.numpy()[0])

            # Take action A and receive reward R and the next state S'
        next_state, reward, done, truncated, info = env.step(action)

            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
        experience_buffer.append((state, action, reward, next_state, done))

        if len(experience_buffer) > 64:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D

                # remove function add it directly
                # selects 5 random tuples from experience storage
            indices = np.random.randint(len(experience_buffer), size=BATCH_SIZE)
                # pulls them out of experience storage into a batch
            batch = [experience_buffer[index] for index in indices]
                # pulls each field in the array into its own array so we can access each individuelle
            experiences = [np.array([experience[field_index] for experience in batch])  # list comprehension
                        for field_index in range(5)]

                # remove function add it directly

            states, actions, rewards, next_states, dones = experiences

            max_next_q = tf.reduce_max(target_q_network(next_states), axis=-1)

            q_targets = rewards + (DISCOUNT * max_next_q * (1 - dones))

                # remove function add it directly
            training_step(q_targets, states, actions, OPTIMIZER, LOSS)
            if t % STEPS_FOR_UPDATE == 0:
                for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
                    target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

        state = next_state.copy()

        if done:
            epsilon = max(EPSILON_MIN, EPSILON_DECAY * epsilon)
            break


