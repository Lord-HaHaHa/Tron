import numpy as np
import tensorflow as tf
import random
import TronGameEngine

# Define your Tron game environment
size_w = 10
size_h = 10
game = TronGameEngine.TronGame(size_w, size_h)
bot = game.registerPlayer((0, 0, 255))
num_actions = 4

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2 + size_w * size_h,)), # x/y Pos + Gamefield
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='linear')
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the exploration strategy
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.999  # Decay rate for exploration rate
discount_factor = 0.7
# Define the replay buffer
replay_buffer = []


# Define the training loop
def train():
    # Sample mini-batch from replay buffer
    indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
    mini_batch = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, dones = zip(*mini_batch)

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # Compute target Q-values
    next_q_values = model.predict(next_states)
    max_next_q_values = np.max(next_q_values, axis=1)
    target_q_values = rewards + discount_factor * max_next_q_values * (1 - dones)

    # Compute current Q-values
    mask = tf.one_hot(actions, num_actions)
    with tf.GradientTape() as tape:
        q_values = model(states)
        q_values = tf.reduce_sum(q_values * mask, axis=1)
        loss = loss_fn(target_q_values, q_values)

    # Update the model
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Training loop
num_episodes = 10_000
batch_size = 10
for episode in range(num_episodes):
    game.reset()
    state = game.getState()
    done = False

    print(f"Training Loop: {episode}")
    while not done:
        # Choose an action
        if np.random.rand() < epsilon:
            action = random.randint(1,4)
        else:
            action = np.argmax(model.predict(np.array([state])))
            print("USE MODEL")

        # Perform the action in the environment
        next_state, reward, done = game.step(action)

        # Store the experience in the replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Update the current state
        state = next_state

        # Train the model
        if len(replay_buffer) >= batch_size:
            train()

        # Decay the exploration rate
        epsilon *= epsilon_decay

# Evaluate the trained model
print("Start the Eval")
total_rewards = 0
num_eval_episodes = 5

for _ in range(num_eval_episodes):
    game.reset()
    state = game.getState()
    done = False

    while not done:
        action = np.argmax(model.predict(np.array([state])))
        print(f'Eval: Model action:{action}')
        next_state, reward, done = game.step(action)
        total_rewards += reward
        state = next_state

average_reward = total_rewards / num_eval_episodes
print("Average reward:", average_reward)
