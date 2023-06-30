from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import abc
import tensorflow as tf
import numpy as np
import TronGameEngine as TGE
import tf_agents
import os

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

configname = 'Model_500'
tempdir = os.path.join('Saves', configname)
policy_dir = os.path.join(tempdir, 'policy')
enemy = False

class TronGameEnv(py_environment.PyEnvironment):
    def __init__(self):
        screen_width = 30
        screen_height = 30
        self.learningType = 3
        self.game_engine = TGE.TronGame(screen_width,screen_height, useTimeout=False, learingType=self.learningType)
        self.player = self.game_engine.registerPlayer((0, 0, 255))
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(screen_width*screen_height,), dtype=np.int32, minimum=0, name='observation')
        self._state = self.game_engine.getState(type=self.learningType)
        self._episode_ended = False
        self.enemy = enemy
        if self.enemy:
            self.enemy_player = self.game_engine.registerPlayer((255, 0, 0))
            self.enemy_pol = tf.saved_model.load(policy_dir)
            #self.tf_py_env = tf_py_environment.TFPyEnvironment(self)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game_engine.reset()
        self._state = self.game_engine.getState(type=self.learningType)
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _render(self):
        self.game_engine.render()

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Let the enemy chose his move
        if self.enemy and len(self.game_engine.act_players) > 1:
#            time_step = self.tf_py_env.current_time_step()
            gamefield_enemy = self.game_engine.getState(type=self.learningType, playerID=self.enemy_player).reshape(1,100)

            ts_enemy = tf_agents.trajectories.TimeStep(observation=tf.convert_to_tensor(gamefield_enemy, dtype=np.int32),
                                               reward=tf.convert_to_tensor([0.0], dtype=np.float32),
                                               discount=tf.convert_to_tensor([0.0], dtype=np.float32),
                                               step_type=tf.convert_to_tensor([0], dtype=np.int32))

            action_step = self.enemy_pol.action(ts_enemy)
            enemy_action = action_step.action
            self.game_engine.registerAction(self.enemy_player, enemy_action)

        if action < 0 or action > 3:
            raise ValueError("'action' should be between 0 and 3")
        else:
            self._state, reward, self._episode_ended = self.game_engine.step(action)

        if self._episode_ended:
            if self.enemy:
                if len(self.game_engine.act_players) > 0:
                    if self.game_engine.act_players[0].id == self.player:
                        return ts.termination(np.array(self._state, dtype=np.int32), reward)
                    else:
                        return ts.termination(np.array(self._state, dtype=np.int32), reward)
                return ts.termination(np.array(self._state, dtype=np.int32), reward)
            else:
                return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            if len(self.game_engine.act_players) > 0:
                if self.game_engine.act_players[0].id != self.player:
                    return ts.termination(np.array(self._state, dtype=np.int32), reward)

            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.0, discount=0.98)


#utils.validate_py_environment(env, episodes=5)