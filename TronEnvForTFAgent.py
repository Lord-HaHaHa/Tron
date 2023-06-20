from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import TronGameEngine as TGE

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class TronGameEnv(py_environment.PyEnvironment):
    def __init__(self):
        screen_width = 30
        screen_height = 30
        self.game_engine = TGE.TronGame(screen_width,screen_height, useTimeout=False, learingType=1)
        self.player = self.game_engine.registerPlayer((0, 0, 255))
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2+screen_width*screen_height,), dtype=np.int32, minimum=-1, name='observation')
        self._state = self.game_engine.getState()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.game_engine.reset()
        self._state = self.game_engine.getState()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action < 0 or action > 3:
            raise ValueError("'action' should be between 0 and 3")
        else:
            self._state, reward, self._episode_ended = self.game_engine.step(action)

        if self._episode_ended:
           return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)

env = TronGameEnv()
utils.validate_py_environment(env, episodes=5)