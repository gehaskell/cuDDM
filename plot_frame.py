import numpy as np
import matplotlib.pyplot as plt

with open("/home/ghaskell/projects_Git/cuDDM/data/data2.txt") as f:
        for x in range(3):
                data = np.array([float(n) for n in f.readline().split()])
                print(data[25])
                size = int(np.sqrt(data.size))
                print(size) 
                data.resize(size,size)
                data[0][0] = 0
                plt.pcolor(data)
                plt.show()

