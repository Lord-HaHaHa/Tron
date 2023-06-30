import TronGameEngine
from pynput import keyboard
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
import tf_agents
import os

# setup
configname = 'Model_30x30'
tempdir = os.path.join('Saves', configname)
policy_dir = os.path.join(tempdir, 'policy')
SCREEN_HEIGHT = 30
SCREEN_WIDTH = 30
game = TronGameEngine.TronGame(SCREEN_HEIGHT, SCREEN_WIDTH)
enemy_player = game.registerPlayer((255, 0, 0))
enemy_pol = tf.saved_model.load(policy_dir)
#tf_py_env = tf_py_environment.TFPyEnvironment(game)


def on_key_press(key):
    #print("key pressed")
    try:
        if key.char == 'w':
            game.registerAction(p1, 3)
        elif key.char == 'a':
            game.registerAction(p1,1)
        elif key.char == 's':
            game.registerAction(p1, 2)
        elif key.char == 'd':
            game.registerAction(p1, 0)
    except AttributeError:
        pass

p1 = game.registerPlayer((0,0,255))
# agame.registerPlayer()
print(f'Player1 ID: {p1}')
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

#p2 = game.registerPlayer()
#print(f'Player2 ID: {p2}')

gameover = False
counter = 0
# game loop
while not gameover:
    # Get Player Movement
    # use Eventlistener for p1
    # Do GameStep
    #counter +=1
    #if counter%10==0:
        # Register Bot Move
        #time_step = tf_py_env.current_time_step()

    gamefield_enemy = game.getState(type=3, playerID=enemy_player).reshape(1, SCREEN_HEIGHT*SCREEN_WIDTH)
    ts_enemy = tf_agents.trajectories.TimeStep(observation=tf.convert_to_tensor(gamefield_enemy, dtype=np.int32),
                                               reward=tf.convert_to_tensor([0.0], dtype=np.float32),
                                               discount=tf.convert_to_tensor([0.0], dtype=np.float32),
                                               step_type=tf.convert_to_tensor([0], dtype=np.int32))
    #ts_enemy = ts.transition(np.array(gamefield_enemy, dtype=np.int32), reward=0.0, discount=0.98)

    action_step = enemy_pol.action(ts_enemy)
    enemy_action = action_step.action
    game.registerAction(enemy_player, enemy_action)

        # game.registerPlayer()
    state, reward, gameover = game.game_step()
    if gameover:
        if game.act_players[0].id == enemy_player:
            counter -= 1
        else:
            counter += 1
        print(f'Win / Loss: {counter}')
        game.reset()
        gameover = False
exit(0)
listener.join()