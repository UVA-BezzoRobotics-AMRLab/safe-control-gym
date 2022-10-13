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
import numpy as np

from collections import deque
import math as m
import visGraph
import networkx as nx
import astar2D as ass
from scipy.interpolate import interp1d

def posToMap(pos,origin,omap,res):
    mapPos = np.round((pos-origin)/res)
    return mapPos
def mapToPos(coord,origin,res):
    pos = coord*res+origin
    return pos
def dist(a,b):
    x1,y1 = a
    x2,y2 = b
    return ((x1-x2)**2+(y1-y2)**2)**0.5

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # Test import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory


class Controller():
    """Template controller class.
    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
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
        # Save environment and conrol parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

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
        mapSize = 40
        omap = np.zeros((mapSize,mapSize))
        inflate = 1 #Use this to inflate obstacles
        gflate = 3 #Use this to inflate gates
        angleFlate = m.pi/8
        # print(self.initial_obs)
        # position = np.array([self.initial_obs[i] for i in (0,2,4)])
        position = np.array([self.initial_obs[0],self.initial_obs[2],1])
        res = 0.15
        origin = np.array([0-mapSize/2*res,0-mapSize/2*res])
        posMap = posToMap(position[:2],origin,omap,res)
        obstacles = np.array(self.NOMINAL_OBSTACLES)
        gates = np.array(self.NOMINAL_GATES)
        #Put obs on map
        omap = ass.put_obs_on_map(obstacles, omap, origin, res, inflate, mapSize)   
        #Put gates on map
        omap,goalList = ass.inflate_gates(gates, omap, gflate, angleFlate, origin, res)
        #get all obstacle positions
        #Identify edges (LONGEST PART OF CODE)
        vertices = np.array([[x,y] for x in range(mapSize) for y in range(mapSize)])
        print("im here 1")
        visEdges = ass.edge_creation(vertices, omap)
        print('finishedEdges')
        G = nx.from_pandas_edgelist(visEdges,source = 'Source',target = 'Target', edge_attr = 'Weight')
        print("I finished poophead")
        
        #########################
        # REPLACE THIS (START) ##
        #########################
        # Example: use visibility graph to construct edges and astar path plan
        if use_firmware:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]

        waypoints = list([position[:2]])   
        waypoints3D = list([position[:3]])
        mapPath = list([posMap])
        print(waypoints3D)
        for idx, g in enumerate(gates):
            startPos = np.array(waypoints[-1])
            start3D = np.array(waypoints3D[-1])
            posMap = posToMap(startPos,origin,omap,res)
            start = tuple(np.array(posMap))
            height = (1 if g[6] == 0 else 0.525)
            #if g[5] > 0.75 or g[5] < 0:
            goal = np.array(g[:2])
            goal3D = np.concatenate([g[:2],[height]])
            goalMap = posToMap(goal, origin, omap, res)
            end = tuple(np.array(goalMap))
        
            path = nx.astar_path(G,start,end,weight='Weight')
            # path = nx.shortest_path(G,start,end,'Weight')
            arrPath = np.array(path)    
        
            arrPath = arrPath[1:]
            pathPos = np.array([mapToPos(row,origin,res) for row in arrPath])
            pathPos[0] = startPos
            pathPos[-1] = goal
            
            pathPos3D = np.array([np.concatenate([row, [0]]) for row in pathPos])
            zInterp = np.linspace(start3D[-1],goal3D[-1],len(pathPos3D))
            pathPos3D[:,-1] = zInterp
            pathPos3D[0] = start3D
            pathPos3D[-1] = goal3D
        
            waypoints.extend(pathPos)
            waypoints3D.extend(pathPos3D)
            mapPath.extend(arrPath)
            print(waypoints)
# height = (initial_info["gate_dimensions"]["tall"]["height"] if g[6] == 0 else initial_info["gate_dimensions"]["low"]["height"])
        #waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])
        print("im here poophead")
        # # Polynomial fit
        self.waypoints = np.array(waypoints3D)
        t = np.arange(self.waypoints.shape[0])
        deg = 20
        t = np.arange(self.waypoints.shape[0])
        fx = np.poly1d(np.polyfit(t, self.waypoints[:,0], deg))
        fy = np.poly1d(np.polyfit(t, self.waypoints[:,1], deg))
        fz = np.poly1d(np.polyfit(t, self.waypoints[:,2], deg))
        duration = 15
        t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
        self.ref_x = fx(t_scaled)
        self.ref_y = fy(t_scaled)
        self.ref_z = fz(t_scaled)


        if self.VERBOSE:
            # Plot trajectory in each dimension and 3D.
            plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

            # Draw the trajectory on PyBullet's GUI
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
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
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's getting_stated scenario.

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == 20*self.CTRL_FREQ:
            command_type = Command(6)  # notify setpoint stop.
            args = []

        elif iteration == 20*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5 
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 23*self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
            z = 1.5
            yaw = 0.
            duration = 6

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 30*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == 33*self.CTRL_FREQ-1:
            command_type = Command(-1)  # Terminate command to be sent once trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
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
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    @timing_step
    def interStepLearn(self,
                       action,
                       obs,
                       reward,
                       done,
                       info):
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

        pass

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