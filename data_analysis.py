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
        with open(filename) as f:
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

    return q_vector, t_vector, iqt


def i_vs_t(q_vector, tau_vec, iqt):
    fig, ax = plt.subplots()
    ax.set(xlabel=r"Lag time $\tau$", ylabel=r"I(q, $\tau$)")

    count_to_plot = 100
    for idx, q in enumerate(q_vector):
        if idx == 0:
            continue
        ax.plot(tau_vec, iqt[idx], label=f"q = {q}")

        count_to_plot -= 1
        if not count_to_plot:
            break 
        
    ax.legend(loc="upper left")

    plt.show()



if __name__ == '__main__':
    q_vector, t_vector, iqt = read_file("./data/iqt.txt")
    i_vs_t(q_vector, t_vector, iqt)