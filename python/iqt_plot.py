import numpy as np
import matplotlib.pyplot as plt


"""
We write data to file, is easier to analyse data in python.

File structure is given

file-start
[0] <vector of q values>
[1] <vector of t values>
[2] <I(q, t) for 1st q value>
[3] <I(q, t) for 2nd q value>
...
[.] <I(q, t) for last q value>
file-end

"""


def read_file(filename):
    try:
        with open(filename, "r") as f:
            # First 2 
            q_vector = [float(n) for n in f.readline().split()]
            t_vector = [float(n) for n in f.readline().split()]
            iqt = []
            for q in q_vector:
                iqt.append([float(n) for n in f.readline().split()])

            print("File read sucessfully.")
            print("q-vector:", q_vector)
            print("t-vector:", t_vector)

    except FileNotFoundError:
        print("Error reading file, check file exsists / formatting")
        q_vector, t_vector, iqt = [], [], [[]]

    return q_vector, t_vector, np.array(iqt)


def i_vs_t(q_vector, tau_vec, iqt):
    fig, ax = plt.subplots()
    ax.set(xlabel=r"Lag time $\tau$", ylabel=r"I(q, $\tau$)")


    for idx, q in enumerate(q_vector):
        ax.plot(tau_vec, iqt[idx], label=f"q = {q}")

        
    ax.legend(loc="upper left")

    plt.show()

def the_other_one(iqtau, q_vector):
    fig, ax = plt.subplots()
    ax.set(xlabel=r"q", ylabel=r"I(q, $\tau$)")

    for tau in range(1, iqtau.shape[1]):
        # plotting the points
        ax.plot(q_vector, iqtau[:, tau], label=f"$\\tau$ = {tau}")

    ax.legend(loc="upper left")
    plt.show()



if __name__ == '__main__':
    q_vector, t_vector, iqt = read_file("/home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt")
    i_vs_t(q_vector, t_vector, iqt)
    the_other_one(iqt, q_vector)