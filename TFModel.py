import numpy as np
import tensorflow as tf
import random
import TronGameEngine
from logger import plot

# Define your Tron game environment
size_w = 20
size_h = 10
LearningType = 2
game = TronGameEngine.TronGame(size_w, size_h, useTimeout=False, learingType=LearningType)
bot = game.registerPlayer((0, 0, 255))
num_actions = 4

# Get InputShape for Model based on LearingType
if LearningType == 1:
    inputShape = 2 + size_w * size_h
elif LearningType == 2:
    inputShape = 9
else:
    inputShape = 0

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(inputShape,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='linear')
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Use GPU if accesebl
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")

# Define the exploration strategy
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99  # Decay rate for exploration rate
discount_factor = 0.9
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
    next_q_values = model.predict(next_states, verbose=0)
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
num_episodes = 200
batch_size = 16
plot_scores = []
plot_mean_scores = []
total_score = 0
for episode in range(1, num_episodes+1):
    game.reset()
    state = game.getState(LearningType)
    done = False
    score = 0

    rndMove = 0
    modelMove = 0

    print(f"Training Loop: {episode}")
    while not done:
        # Choose an action
        if np.random.rand() < epsilon:
            action = random.randint(1,4)
            rndMove += 1
        else:
            action = np.argmax(model.predict(np.array([state]))) + 1
            modelMove += 1

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

        # Save Score
        score += reward

    # Print Stats after GameOver
    if modelMove != 0:
        print(f"Used Random / ModelMove: {rndMove} / {modelMove}, {epsilon}%")
    else:
        print(f"Used Random / ModelMove: {rndMove} / {modelMove}, inf")

    plot_scores.append(score)
    total_score += score
    mean_score = total_score / episode
    plot_mean_scores.append(mean_score)
    plot(plot_scores, plot_mean_scores)
plot(plot_scores, plot_mean_scores, "LeaningCurve.png")
# Evaluate the trained model
print("Start the Eval")
total_rewards = 0
num_eval_episodes = 5

for _ in range(num_eval_episodes):
    game.reset()
    state = game.getState(LearningType)
    done = False

    while not done:
        action = np.argmax(model.predict(np.array([state]))) + 1
        print(f'Eval: Model action:{action}')
        next_state, reward, done = game.step(action)
        total_rewards += reward
        state = next_state

average_reward = total_rewards / num_eval_episodes
print("Average reward:", average_reward)

# Store Traind model
model.save("TF-Model-Tron")