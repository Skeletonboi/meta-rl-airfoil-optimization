from generateNaca import naca4
import numpy as np

state = np.array(naca4([7.79526363 ,5.81967631, 6.16088396], 50)).transpose()


# WRITE COORDS TO DAT FILE
name = 'runA8'
save_path = f'{name}.dat'
with open(save_path, 'w') as f:
    f.write(f'2D: {name}\n')
    for i in range(state.shape[0]):
        f.write(f'{state[i,0]},{state[i,1]}\n')
