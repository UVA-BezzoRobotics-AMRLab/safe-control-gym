"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 4 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
import os
import sys
import math as m
import numpy as np
import numpy.matlib
import pandas as pd
import astar2D as ass
import networkx as nx
import tensorflow as tf

from collections import deque
from scipy.interpolate import interp1d

try:
    from competition_utils import (
        Command,
        PIDController,
        timing_step,
        timing_ep,
        plot_trajectory,
        draw_trajectory,
    )
except ImportError:
    # Test import.
    from .competition_utils import (
        Command,
        PIDController,
        timing_step,
        timing_ep,
        plot_trajectory,
        draw_trajectory,
    )


def posToMap(pos, origin, omap, res):
    mapPos = np.round((pos - origin) / res)
    return mapPos


def mapToPos(coord, origin, res):
    pos = coord * res + origin
    return pos


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def min_jerk(xi, xf, t):
    d = t[-1]
    N = len(t)
    a = np.matlib.repmat((xf - xi), N, 1)
    bval = np.matlib.repmat(
        10 * (t / d) ** 3 - 15 * (t / d) ** 4 + 6 * (t / d) ** 5, 3, 1
    )
    bval = np.transpose(bval)
    output = np.matlib.repmat(xi, N, 1) + a * bval
    return output


class Controller:
    """Template controller class."""

    def __init__(
        self,
        initial_obs,
        initial_info,
        use_firmware: bool = False,
        buffer_size: int = 100,
        verbose: bool = False,
        env_num: int = 0,
        train: bool = True,
        trainer=None,
        train_count: int = 0,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """

        self.model = trainer
        self.is_train_data = train
        self.epsilon = min(float(train_count)*.25, 6.0)

        self.data = []
        self.df = None
        self.current_goal = 0
        self.env_num = env_num

        # Save environment parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        self.final_goal = initial_info["stabilization_goal"]
        self.goal_tolerance = initial_info["stabilization_goal_tolerance"]

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller ror debugging and test
            # Do NOT use for the IROS 2022 competition.
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        ###### ASTAR STUFF ######
        #########################
        mapSize = 60
        omap = np.zeros((mapSize, mapSize))
        inflate = 1  # Use this to inflate obstacles
        gflate = 3  # Use this to inflate gates
        angleFlate = m.pi / 8
        # print(self.initial_obs)
        # position = np.array([self.initial_obs[i] for i in (0,2,4)])
        position = np.array([self.initial_obs[0], self.initial_obs[2], 1])
        res = 0.15
        origin = np.array([0 - mapSize / 2 * res, 0 - mapSize / 2 * res])
        posMap = posToMap(position[:2], origin, omap, res)
        obstacles = np.array(self.NOMINAL_OBSTACLES)
        gates = np.array(self.NOMINAL_GATES)
        # Put obs on map
        omap = ass.put_obs_on_map(obstacles, omap, origin, res, inflate, mapSize)
        # Put gates on map
        print(gates)
        print(omap)
        print(gflate)
        print(angleFlate)
        print(origin)
        print(res)
        omap, goalList = ass.inflate_gates(gates, omap, gflate, angleFlate, origin, res)
        # get all obstacle positions
        # Identify edges (LONGEST PART OF CODE)
        vertices = np.array([[x, y] for x in range(mapSize) for y in range(mapSize)])
        visEdges = ass.edge_creation(vertices, omap)
        G = nx.from_pandas_edgelist(
            visEdges, source="Source", target="Target", edge_attr="Weight"
        )

        #########################
        # REPLACE THIS (START) ##
        #########################

        if use_firmware:
            waypoints = [
                (
                    self.initial_obs[0],
                    self.initial_obs[2],
                    initial_info["gate_dimensions"]["tall"]["height"],
                )
            ]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [
                (self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])
            ]

        waypoints = list([position[:2]])
        waypoints3D = list([position[:3]])
        mapPath = list([posMap])
        print(waypoints3D)
        for idx, g in enumerate(gates):
            startPos = np.array(waypoints[-1])
            start3D = np.array(waypoints3D[-1])
            posMap = posToMap(startPos, origin, omap, res)
            start = tuple(np.array(posMap, dtype=int))
            height = (
                initial_info["gate_dimensions"]["tall"]["height"]
                if g[6] == 0
                else initial_info["gate_dimensions"]["low"]["height"]
            )
            # if g[5] > 0.75 or g[5] < 0:
            goal = np.array(g[:2])
            goal3D = np.concatenate([g[:2], [height]])
            goalMap = posToMap(goal, origin, omap, res)
            end = tuple(np.array(goalMap, dtype=int))

            path = nx.astar_path(G, start, end, weight="Weight")
            # path = nx.shortest_path(G,start,end,'Weight')
            arrPath = np.array(path)

            arrPath = arrPath[1:]
            pathPos = np.array([mapToPos(row, origin, res) for row in arrPath])
            pathPos[0] = startPos
            pathPos[-1] = goal

            pathPos3D = np.array([np.concatenate([row, [0]]) for row in pathPos])
            zInterp = np.linspace(start3D[-1], goal3D[-1], len(pathPos3D))
            pathPos3D[:, -1] = zInterp
            pathPos3D[0] = start3D
            pathPos3D[-1] = goal3D
            print(pathPos3D)
            waypoints.extend(pathPos[:-1])
            waypoints3D.extend(pathPos3D[:-1])
            mapPath.extend(arrPath)
        waypoints3D.extend([goal3D])

        waypts = np.array(waypoints3D[1:])
        allTraj = [0, 0, 0]
        freq = 1.0 / self.CTRL_FREQ
        move_time = 0.3
        t = np.arange(0, move_time, freq)
        for idx in range(waypts.shape[0] - 1):
            start = waypts[idx]
            end = waypts[idx + 1]
            traj = min_jerk(start, end, t)
            allTraj = np.vstack([allTraj, traj])
        allTraj = allTraj[1:]
        self.waypoints = waypts
        self.ref_x = allTraj[:, 0]
        self.ref_y = allTraj[:, 1]
        self.ref_z = allTraj[:, 2]
        t_scaled = np.linspace(0, len(waypts), len(allTraj))

        # Draw the trajectory on PyBullet's GUI
        draw_trajectory(
            initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z
        )
        self.step = 0

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self, time, obs, reward=None, done=None, info=None):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError(
                "[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False."
            )

        iteration = int(time * self.CTRL_FREQ)
        self.step = iteration

        command_type = Command(0)
        args = []

        #########################
        # REPLACE THIS (START) ##
        #########################
        # Handwritten solution for GitHub's example scenario.
        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        elif iteration < len(self.ref_x):

            pos = np.array([obs[0], obs[1], obs[2]])
            old_target = [
                self.ref_x[iteration - 1],
                self.ref_y[iteration - 1],
                self.ref_z[iteration - 1],
            ]

            if np.linalg.norm(pos - old_target) < self.epsilon+100:
                dat = self.generate_data(obs)
                data_list = [dat[key] for key in dat.keys()]
                all_data = data_list[:36]
                all_data = tf.expand_dims(tf.stack([all_data, all_data, all_data]), axis=0)
                inputs = {"state": all_data[:,:,:16], "env":all_data[:,:,16:]}
                output = self.model.inference(inputs)
                target_pos = self.model
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
            else:
                target_pos = np.array(
                    [
                        self.ref_x[iteration],
                        self.ref_y[iteration],
                        self.ref_z[iteration],
                    ]
                )
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)

            target_yaw = 0
            target_rpy_rates = np.zeros(3)

            # command_type = Command(5)
            # args = [target_pos, 0, self.CTRL_FREQ, False]
            command_type = Command(1)
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self, time, obs, reward=None, done=None, info=None):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the IROS 2022 Safe Robot Learning competition.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError(
                "[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True."
            )

        iteration = int(time * self.CTRL_FREQ)
        self.step = iteration

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array(
                [self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]]
            )
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def generate_data(self, obs):
        if self.step > len(self.ref_x) - 5:
            return None

        end = min(self.step + 10, len(self.ref_x))
        future_poses = [
            self.ref_x[self.step : end],
            self.ref_y[self.step : end],
            self.ref_z[self.step : end],
        ]

        if self.step > len(self.ref_x) - 10:
            future_poses[0] = np.pad(
                future_poses[0], (0, 10 - (len(self.ref_x) - self.step)), mode="edge"
            )
            future_poses[1] = np.pad(
                future_poses[1], (0, 10 - (len(self.ref_y) - self.step)), mode="edge"
            )
            future_poses[2] = np.pad(
                future_poses[2], (0, 10 - (len(self.ref_z) - self.step)), mode="edge"
            )

        quad_pos = np.array([obs[0], obs[2], obs[4]])
        curr_goal = None

        if self.current_goal < len(self.NOMINAL_GATES):
            curr_goal = self.NOMINAL_GATES[self.current_goal][:6]
        else:
            curr_goal = [
                self.final_goal[0],
                self.final_goal[1],
                self.final_goal[2],
                0,
                0,
                0,
            ]

        if np.linalg.norm(quad_pos - np.array(curr_goal[:3])) < self.goal_tolerance:
            self.current_goal += 1

        if self.current_goal < len(self.NOMINAL_GATES):
            curr_goal = self.NOMINAL_GATES[self.current_goal][:6]
        else:
            curr_goal = [
                self.final_goal[0],
                self.final_goal[1],
                self.final_goal[2],
                0,
                0,
                0,
            ]

        obstacle_list = self.NOMINAL_OBSTACLES
        obstacle_list = [[ob[0], ob[1]] for ob in obstacle_list]

        while len(obstacle_list) < 10:
            obstacle_list.append([-10000, -10000])

        obstacle_list = np.array(obstacle_list)
        obstacle_dists = np.linalg.norm(quad_pos[:2] - obstacle_list, axis=1)
        obstacle_list = obstacle_list[np.argsort(obstacle_dists)]

        rot_mat = euler2Mat(obs[6], obs[7], obs[8])
        rot_mat_flat = rot_mat.flatten()
        vel = np.array([obs[1], obs[3], obs[5]])
        err_trans = np.array(curr_goal[:3]) - quad_pos
        err_rot = angleBetweenMats(
            rot_mat, euler2Mat(curr_goal[3], curr_goal[4], curr_goal[5])
        )

        step_data_dir = {
            "vel_x": vel[0],
            "vel_y": vel[1],
            "vel_z": vel[2],
            "R_0": rot_mat_flat[0],
            "R_1": rot_mat_flat[1],
            "R_2": rot_mat_flat[2],
            "R_3": rot_mat_flat[3],
            "R_4": rot_mat_flat[4],
            "R_5": rot_mat_flat[5],
            "R_6": rot_mat_flat[6],
            "R_7": rot_mat_flat[7],
            "R_8": rot_mat_flat[8],
            "err_x": err_trans[0],
            "err_y": err_trans[1],
            "err_z": err_trans[2],
            "err_theta": err_rot,
        }

        axes = ["x", "y", "z"]
        for i, ob in enumerate(obstacle_list):

            if i > 9:
                break

            for j, val in enumerate(ob):
                step_data_dir[f"obs_{i}_{axes[j]}"] = val

        for m in range(3):
            for i in range(10):
                for ind, ax in enumerate(axes):
                    step_data_dir[f"traj{m}_{i}{ax}"] = future_poses[ind][i]

        return step_data_dir

    @timing_step
    def interStepLearn(self, action, obs, reward, done, info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################

        dat = self.generate_data(obs)

        if dat != None:
            self.data.append(dat)

        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        self.df = pd.DataFrame(self.data)
        # self.df.index.name="steps"
        train_str = "train" if self.is_train_data else "val"
        self.df.to_csv(f"./planner_model/data/{train_str}/env_{self.env_num}.dat")

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0


def euler2Mat(roll, pitch, yaw):

    R_theta = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R_pitch = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    R_mat = np.matmul(R_theta, np.matmul(R_pitch, R_yaw))

    return R_mat


# Courtesy of https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
def angleBetweenMats(P, Q):
    R = np.dot(P, Q.T)
    theta = (np.trace(R) - 1) / 2
    return np.arccos(theta)
