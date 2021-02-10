import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

with open("/home/ghaskell/projects_Git/cuDDM/data/data2.txt") as f:
        for x in range(1):
                data = np.array([float(n) for n in f.readline().split()])
                print(data[25])
                size = int(np.sqrt(data.size))
                print(size) 

                min_ = data.min()
                max_ =  data.max()

                data.resize(size,size)
                data[0][0] = 0

                fig, ax = plt.subplots()
                ax.pcolor(data)#, norm=colors.LogNorm(vmin=min_, vmax=max_), cmap='PuBu_r')
                plt.show()

