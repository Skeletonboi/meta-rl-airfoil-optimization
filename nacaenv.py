import sys
import os
import numpy as np
import pandas as pd
import gym
import time
from gym import spaces
sys.path.append('./')
sys.path.append('./eval/')
from eval.SOLVE import XFOILmod
from eval.POST import postProcess
from generateNaca import naca4

class nacaEnv(gym.Env):
    def __init__(self, args, paths, runName, train=True):
        super(nacaEnv, self).__init__()


        self.max_steps_env = args[0]
        self.nPoints = args[1]
        self.Vinf = args[2]
        self.AOA = args[3]
        self.Ma = args[4]
        self.Re = args[5]
        self.train = train

        self.iter = 0
        self.paths = paths
        self.fin_dir = paths[2]
        self.runName = runName
        
        self.actionMins = [-3,-3,-10]
        self.actionMaxs = [3,3,10]
        self.stateMins = [0,0,5]
        self.stateMaxs = [9.5,9,20]

        self.done = False

        self.action_space = spaces.Box(low=np.array(self.actionMins), high=np.array(self.actionMaxs), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array(self.stateMins), high=np.array(self.stateMaxs), dtype=np.float32)

        self.recurse_counter = 0
        self.recurse_limit = 100
        self.cd0 = 0

    def reset(self):
        self.iter = 0
        self.done = False

        state = []
        for i in range(len(self.stateMaxs)):
            mean = (self.stateMaxs[i]+self.stateMins[i])/2
            state.append(np.clip(np.random.normal(mean, mean/10), self.stateMins[i], self.stateMaxs[i])) 
        self.state = state

        # self.state = [4.75, 4.5, 10]

        initialGeom = np.array(naca4(self.state, self.nPoints)).transpose()
        initialStatePath = self.writeDAT(initialGeom, 'initialNACA')
        initCL, initCD, didCrash = self.solveFinCD(initialStatePath, plot=False)
        if (initCD == 0) or (didCrash):
            if initCD == 0:
                self.cd0 += 1
            self.recurse_counter += 1
            if self.recurse_counter > self.recurse_limit:
                print('YO?', self.cd0)
                assert(1==0)
            print('Initial State Failed, recursing ...')
            return self.reset()
        self.recurse_counter = 0
        self.initialCLCD = initCL/initCD
        return self.state

    def step(self, action):
        self.iter += 1
        # Normalize actions (?)
        n_state = self.state + action
        n_state = self.boundFinState(n_state)
        # generate geometry points:
        geometry = np.array(naca4(n_state, self.nPoints)).transpose()
        # write geometry to DAT file
        name = f'iter{self.iter}'
        finPath = self.writeDAT(geometry, name)
        print('Fin Path',finPath)
        # Solve for CL/CD ratio
        newCL, newCD, didCrash = self.solveFinCD(finPath, plot=False)
        if (newCD == 0) or didCrash:
            self.done = True
            reward = -5
            ret_CLCD = -5
        else:
            reward = max(newCL/newCD - self.initialCLCD, -5)
            ret_CLCD = newCL/newCD
            # reward = newCL/newCD
        # update state, return new reward(CD)
        self.state = n_state
        # if reward < 0:
        #     return reward-3
        print('RETURNING REWARD ', reward, ' crash/no-converge: ', didCrash)
        if self.iter == self.max_steps_env:
            self.done = True
        if not self.train:
            return n_state, reward, self.done, ret_CLCD
        else:
            return n_state, reward, self.done, {}

    def boundFinState(self, state):
        # Ensures action-updated state is within bounds
        for i in range(len(state)):
            state[i] = max(self.stateMins[i], min(self.stateMaxs[i], state[i]))
        return state

    def writeDAT(self, state, name):
        # WRITE COORDS TO DAT FILE
        save_path = os.path.join(self.fin_dir, f'{name}.dat')
        with open(save_path, 'w') as f:
            f.write(f'2D: {name}\n')
            for i in range(state.shape[0]):
                f.write(f'{state[i,0]},{state[i,1]}\n')
        return save_path

    def solveFinCD(self, finPath, plot):
        # SOLVE XFOIL
        # Freestream velocity [] (just leave this at 1)
        # Angle of attack [deg]
        # Convert angle of attack to radians
        AoAR = self.AOA*(np.pi/180)                  # Angle of attack [rad]
        if plot:
            plotFlags = [0,0,1,1,0,0]
        else:
        # Plotting flags
            plotFlags = [0,      # Airfoil with panel normal vectors
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
        xFoilResults, didCrash = XFOILmod(PPAR, self.AOA, self.Ma, self.Re, finPath, self.runName)
        try:
            CLtuple, CD, _ = postProcess(xFoilResults, plotFlags, AoAR, self.Vinf)
        except:
            return 0, 0, True
        print('Computed Results: ', CLtuple, CD)
        CL = CLtuple[2]
        CD = max(0, CD)
        return CL, CD, didCrash

    def render(self):
        pass
    