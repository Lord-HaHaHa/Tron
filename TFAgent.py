from __future__ import absolute_import, division, print_function

import base64

import TronEnvForTFAgent
import IPython
import matplotlib.pyplot as plt

import numpy as np
import PIL.Image

import tensorflow as tf

import logger
import os
import tempfile
from IPython import display
from tf_agents.policies import policy_saver
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import categorical_q_network

configname = 'Model_30x30'

use_trainedPol = False
tempdir = os.path.join('Saves', configname)

num_iterations = 40_000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1# @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}

gamma = 0.99
num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

batch_size = 128  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 15  # @param {type:"integer"}
eval_interval = 500  # @param {type:"integer"}

# Logging
plot_path = 'LearningCurve500Node.png'
all_avg_return = 0
num_avg_return = 1
record = -100

# Init Env
env = TronEnvForTFAgent.TronGameEnv()
train_py_env = TronEnvForTFAgent.TronGameEnv()
eval_py_env = TronEnvForTFAgent.TronGameEnv()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Sepecs for the NN
fc_layer_params = (500,)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

# Define the NN
categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

# Define the Agent
agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

# Define Policy's
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            print(time_step.is_last())
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# Define Replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

# Define Checkpointer
checkpoint_dir = os.path.join(tempdir, 'checkpoint')
global_step = tf.compat.v1.train.get_or_create_global_step()

train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )

train_checkpointer.initialize_or_restore()

# Setup Policies Saver
policy_dir = os.path.join(tempdir, 'policy')
tf_policy_saver = policy_saver.PolicySaver(agent.policy)

print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Reset the environment.
time_step = train_py_env.reset()


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)

# Collect Initial Steps
for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)


def plot(scores, avg, save_path=None):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(avg)
    plt.show(block=False)

    if save_path:
        plt.savefig(save_path)


all_avg_return = 0
num_avg_return = 1
record = -100
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
all_avg_return += avg_return
record = avg_return
returns = [avg_return]
avg_returns = [all_avg_return / num_avg_return]

# Training Loop
for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        num_avg_return += 1
        all_avg_return += avg_return
        print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
        avg_returns.append(all_avg_return / num_avg_return)
        returns.append(avg_return)
        logger.plot(returns, avg_returns, plot_path)
        print(f'Record: {record}, AVGReturn:{avg_return}')
        if(avg_return > record):
            record = avg_return
            print("New Record")
            tf_policy_saver = policy_saver.PolicySaver(agent.policy)
            tf_policy_saver.save(policy_dir)
            train_checkpointer.save(global_step)


exit(0)