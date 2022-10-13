import os

import numpy as np
import tensorflow as tf

from tqdm import tqdm

# from nets import create_network
from tensorflow.keras import Model


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def call(self, x):
        return self._internal_call(x)

    def _create(self):
        raise NotImplementedError

    def _internal_call(self):
        raise NotImplementedError


class PlanLearner(Network):
    def __init__(self):
        super(PlanLearner, self).__init__()

        self.obs_embedding = [
            tf.keras.layers.Conv1D(
                int(128.0), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(64), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(34), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(34), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
        ]

        # shuold probably change padding to 'valid' (which is no padding)
        self.resize_op_obs = [
            tf.keras.layers.Conv1D(
                3, kernel_size=3, strides=1, padding="valid", dilation_rate=1
            )
        ]

        self.state_embedding = [
            tf.keras.layers.Conv1D(
                int(64.0), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(34), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(34), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(34), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
        ]

        # shuold probably change padding to 'valid' (which is no padding)
        self.resize_op_state = [
            tf.keras.layers.Conv1D(
                3, kernel_size=3, strides=1, padding="valid", dilation_rate=1
            )
        ]

        # output size is state_dim*out_length + 1
        # The +1 is the collision cost of the trajectory
        self.plan_model = [
            tf.keras.layers.Conv1D(
                int(64.0), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(128.0), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(
                int(128.0), kernel_size=2, strides=1, padding="same", dilation_rate=1
            ),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Conv1D(31, kernel_size=1, strides=1, padding="same"),
        ]

    def _state_embedding(self, _input):
        x = _input
        for f in self.state_embedding:
            x = f(x)
        x = tf.transpose(x, (0, 2, 1))

        for f in self.resize_op_state:
            x = f(x)
        x = tf.transpose(x, (0, 2, 1))
        return x

    def _env_embedding(self, _input):
        x = _input
        for f in self.obs_embedding:
            x = f(x)
        x = tf.transpose(x, (0, 2, 1))

        for f in self.resize_op_obs:
            x = f(x)
        x = tf.transpose(x, (0, 2, 1))
        return x

    def _plan(self, _input):
        x = _input
        for f in self.plan_model:
            x = f(x)
        return x

    def _internal_call(self, inputs):

        state_obs = inputs["state"]
        env_obs = inputs["env"]

        st_em = self._state_embedding(state_obs)
        en_em = self._env_embedding(env_obs)
        con_em = tf.concat((st_em, en_em), axis=-1)

        output = self._plan(con_em)
        return output


if __name__ == "__main__":
    planner = PlanLearner()

    x = {"state": np.zeros([8, 3, 16]), "env": np.zeros([8, 3, 60])}
    x = planner(x)

    print(x.shape)
