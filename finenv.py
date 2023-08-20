import sys
import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
sys.path.append('./')
sys.path.append('./eval/')
from eval.SOLVE import XFOILmod
from eval.POST import postProcess

class finEnv(gym.Env):
    def __init__(self, args, fin_dir, runName):
        super(finEnv, self).__init__()
        nPoints = args[0]
        self.Vinf = args[1]
        self.AOA = args[2]
        self.Ma = args[3]
        self.Re = args[4]

        self.iter = 0
        self.fin_dir = fin_dir
        self.runName = runName
        self.nPoints = nPoints
        self.length = self.nPoints*2+3
        self.absYMin = 0.005
        self.absYMax = 0.1
        self.actionMax = 1
        self.state = self.reset() # Initialize state
        self.done = False

        # self.action_space = spaces.Box(low=np.array([0.0]*nPoints+[-1.0]*nPoints), high=np.array([1.0]*nPoints+[0.0]*nPoints), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.actionMax, high=self.actionMax, shape=(nPoints*2,))
        # self.observation_space = spaces.Box(low=np.array([0.0]*nPoints+[-1.0]*nPoints), high=np.array([1.0]*nPoints+[0.0]*nPoints), dtype=np.float32)
        obsLowBound_x = np.array([0]*self.length)
        obsHighBound_x = np.array([1]*self.length)
        obsLowBound_y = np.array([self.absYMin]*(nPoints+2) + [-self.absYMax]*nPoints + [0])
        obsHighBound_y = np.array([self.absYMax]*(nPoints+2) + [-self.absYMin]*nPoints + [0])
        lowBound = np.concatenate((obsLowBound_x, obsLowBound_y)).reshape(self.length,2)
        highBound = np.concatenate((obsHighBound_x, obsHighBound_y)).reshape(self.length,2)
        self.observation_space = spaces.Box(low=lowBound, high=highBound, dtype=np.float32)

    def reset(self):
        self.iter = 0
        self.done = False
        # Space X coordinates evenly, Y coordinates randomly distributed within expected range
        x_half = np.linspace(1,0,self.nPoints+2)
        x = np.concatenate((x_half, x_half[::-1][1:])).reshape(self.length,1)

        # y_1half = np.random.uniform(low=self.absYMin, high=self.absYMax, size=(self.nPoints,))
        # y_2half = np.random.uniform(low=-self.absYMax, high=-self.absYMin, size=(self.nPoints,))
        y_1half = np.random.normal(self.absYMax/2, self.absYMax/10, size=(self.nPoints,))
        y_2half = np.random.normal(-self.absYMax/2, self.absYMax/10, size=(self.nPoints,))
        y = np.concatenate(([0], y_1half, [0], y_2half, [0])).reshape(self.length,1)
        state = np.concatenate((x,y),1)

        self.state = state
        return state

    def step(self, action):
        # State-space = (self.nPoints/2)+3 x 2 matrix, continuous # The +2 are the fin leading and tail edges that are kept 
        # Action-space = (self.nPoints/2) x 2 matrix, continuous, y-coord delta to non-leading/tail points
        self.iter += 1
        # Normalize actions (?)
        action /= 100
        x_state = self.state[:,0].reshape(self.length,1)
        y_1state = self.state[:,1][1:self.nPoints+1] + action[:self.nPoints]
        y_2state = self.state[:,1][self.nPoints+2:self.nPoints*2+2] + action[self.nPoints:]
        y_state = np.concatenate(([0],y_1state,[0],y_2state,[0])).reshape(self.length,1)
        n_state = np.concatenate((x_state,y_state),1)

        n_state = self.fixFinGeometry(n_state)
        print(n_state)
        # write new state to dat file
        name = f'iter{self.iter}'
        finPath = self.writeDAT(n_state, name)
        print('Fin Path',finPath)
        # Solve for CL/CD ratio
        reward, didCrash = self.solveFinCD(finPath, self.Vinf, self.AOA, self.Ma, self.Re, False)
        # update state, return new reward(CD)
        self.state = n_state
        # if reward < 0:
        #     return reward-3
        print('RETURNING REWARD ', reward, ' crash/no-converge: ', didCrash)
        if self.iter == 20:
            self.done = True
        elif didCrash:
            self.done = True
            reward = -10
        return n_state, reward, self.done, {}

    def fixFinGeometry(self, state):
        # Ensures action-updated state creates relatively well-behaved geometry
        for i in range(self.nPoints):
            state[i+1,1] = max(self.absYMin, min(self.absYMax, state[i+1,1]))
            state[i+self.nPoints+2,1] = max(-self.absYMax, min(-self.absYMin, state[i+self.nPoints+2,1]))
            # if state[i+1,1] < self.absYMin:
            #     state[i+1,1] = self.absYMin
            # if state[i+self.nPoints+2,1] > -self.absYMin:
            #     state[i+self.nPoints+2,1] = -self.absYMin
        return state

    def writeDAT(self, state, name):
        # WRITE COORDS TO DAT FILE
        save_path = os.path.join(self.fin_dir, f'{name}.dat')
        with open(save_path, 'w') as f:
            f.write(f'2D: {name}\n')
            for i in range(state.shape[0]):
                f.write(f'{state[i,0]},{state[i,1]}\n')
        return save_path

    def solveFinCD(self, finPath, Vinf, AoA, Ma, Re, plot):
        # SOLVE XFOIL
        # Flag to specify creating or loading airfoil
        NACA = '2412'
        isNACA = False
        plot = False
        # User-defined knowns
        # Vinf = 1                                 # Freestream velocity [] (just leave this at 1)
        # AoA  = 0                                 # Angle of attack [deg]
        # Ma = 0.3
        # Re = 3500000
        # finPath = "./2412_12.dat"
        # Convert angle of attack to radians
        AoAR = AoA*(np.pi/180)                  # Angle of attack [rad]
        
        if plot:
            flagPlot = [0,0,1,1,0,0]
        else:
        # Plotting flags
            flagPlot = [0,      # Airfoil with panel normal vectors
                        0,      # Geometry boundary pts, control pts, first panel, second panel
                        0,      # Cp vectors at airfoil surface panels
                        0,      # Pressure coefficient comparison (XFOIL vs. VPM)
                        0,      # Airfoil streamlines
                        0]      # Pressure coefficient contour

        # %% XFOIL - CREATE/LOAD AIRFOIL
        # PPAR menu options
        PPAR = ['170',                                                                  # "Number of panel nodes"
                '4',                                                                    # "Panel bunching paramter"
                '1',                                                                    # "TE/LE panel density ratios"
                '1',                                                                    # "Refined area/LE panel density ratio"
                '1 1',                                                                  # "Top side refined area x/c limits"
                '1 1']                                                                  # "Bottom side refined area x/c limits"
        # Call XFOIL function to obtain the following:
        # - Airfoil coordinates
        # - Pressure coefficient along airfoil surface
        # - Lift, drag, and moment coefficients
        # xFoilResults = XFOIL(NACA, PPAR, AoA, flagAirfoil)
        xFoilResults, didCrash = XFOILmod(NACA, PPAR, AoA, Ma, Re, isNACA, finPath, self.runName)
        if didCrash:
            return 0, didCrash
        try:
            CLtuple, CD, _ = postProcess(xFoilResults, flagPlot, AoAR, Vinf)
        except:
            return -10, True
        print('Final Results: ', CLtuple, CD)
        CL = CLtuple[2]
        if CD == 0:
            if CL == 0:
                return -5, didCrash
            else:
                return -5, didCrash
        return CL/CD, didCrash

    def render(self):
        pass
    