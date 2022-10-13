import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# tf.config.run_functions_eagerly(True)
from tensorflow.keras.losses import Loss

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


class RWTALoss(Loss):
    def __init__(self, T=0.1, modes=3):

        super(RWTALoss, self).__init__()
        self.space_loss = DiscretePositionLoss()

    def call(self, y_true, y_pred):

        mode_losses = []
        pred_costs = []

        for k in range(3):
            pred_len = y_pred.shape[-1]
            pred_cost = tf.reshape(y_pred[:, k, 0], (-1, 1))
            pred_traj = tf.reshape(y_pred[:, k, 1:], (-1, pred_len - 1))
            mode_loss = []

            for i in range(y_true.shape[1]):
                mode_loss.append(self.space_loss(y_true[:, i], pred_traj))

            mode_loss = tf.concat(mode_loss, axis=1)
            mode_loss = tf.expand_dims(mode_loss, axis=-1)
            pred_costs.append(pred_cost)
            mode_losses.append(mode_loss)

        mode_losses = tf.concat(mode_losses, axis=-1)
        max_idx = tf.argmin(mode_losses, axis=-1)  # [B,K]
        epsilon = 0.01
        loss_matrix = tf.zeros_like(pred_costs)
        for k in range(y_true.shape[1]):
            selection_matrix = tf.one_hot(max_idx[:, k], depth=3)
            loss_matrix = loss_matrix + (
                selection_matrix * mode_losses[:, k, :] * (1 - epsilon)
            )

        # considering all selected modes over all possible gt trajectories
        final_selection_matrix = tf.cast(
            tf.greater(loss_matrix, 0.0), tf.float32
        )  # [B,M]
        # give a cost to all trajectories which received no vote

        relaxed_cost_matrix = (
            (1.0 - final_selection_matrix) * mode_losses[:, 0, :] * epsilon / 2.0
        )
        final_cost_matrix = loss_matrix + relaxed_cost_matrix
        trajectory_loss = tf.reduce_mean(tf.reduce_mean(final_cost_matrix, axis=-1))
        return trajectory_loss

class DiscretePositionLoss(Loss):
    def __init__(self):
        super(DiscretePositionLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):

        average_loss = self.mse_loss(y_true, y_pred)
        average_loss = tf.expand_dims(average_loss, axis=-1)

        # For some reason *10 helps with optimization acc. to scaramuzza group
        final_loss = average_loss * 10
        return final_loss


class TrajectoryCostLoss(Loss):
    def __init__(self):
        super(TrajectoryCostLoss, self).__init__()
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def call(self, data, y_pred):

        obstacle_dat = data[0]
        states = data[1]

        batch_size = y_pred.shape[0]
        traj_costs = []
        pred_costs = y_pred[:, :, 0]

        for k in range(batch_size):
            traj_costs.append(
                tf.stop_gradient(
                    self._compute_traj_cost(
                        obstacle_dat[k], states[k], y_pred[k, :, 1:]
                    )
                )
            )

        traj_costs = tf.stack(traj_costs)
        traj_cost_loss = 2 * self.mse_loss(traj_costs, pred_costs)
        return traj_cost_loss

    def _compute_traj_cost(self, obstacle_dat, state, pred_trajectories):

        num_modes = pred_trajectories.shape[0]
        costs = []
        for k in range(num_modes):
            traj_cost = tf.py_function(
                func=self._compute_single_traj_cost,
                inp=[obstacle_dat, state, pred_trajectories[k]],
                Tout=tf.float32,
            )

            traj_cost.set_shape((1,))
            costs.append(traj_cost)

        costs = tf.stack(costs)
        return costs

    def _compute_single_traj_cost(self, obstacle_dat, state, trajectory):

        quadrotor_size = 0.1
        collision_threshold = 0.3

        state = state.numpy()
        traj_np = trajectory.numpy()
        traj_len = traj_np.shape[0]

        traj_np = np.reshape(traj_np, ((-1, traj_len)))
        traj_np = traj_np.reshape(10, 3)[:, :2]
        traj_len = traj_np.shape[0]

        cost = 0.0
        for i in range(traj_len):
            dists = get_dist_vector(obstacle_dat, traj_np[i], collision_threshold)

            if len(dists) > 0:
                dist = np.sqrt(np.min(dists))

                if dist < quadrotor_size:
                    cost += -2.0 / (quadrotor_size**2) * dist**2 + 4.0
                else:
                    cost += (
                        2
                        * (quadrotor_size - dist)
                        / (collision_threshold - quadrotor_size)
                        + 2
                    )

        cost = cost / traj_len
        cost = np.array(cost, dtype=np.float32).reshape((1,))
        return cost

def get_dist_vector(obstacle_dat, state, collision_threshold):

    obs = obstacle_dat[0].numpy().reshape(10, 2)

    dists = np.linalg.norm(obs - state, axis=1)
    dists = dists[(dists < collision_threshold)]

    return dists
