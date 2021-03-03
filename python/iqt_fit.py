import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


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


def dampend_osc(x, ampl, freq, damp, offset):
    return ampl * (1 - np.exp(np.cos(freq * x))) * np.exp(-damp * x) + offset

def fit(iqtau, q_vector, tau_vector):

    # #  preliminary search for qstart
    # # |    _
    # #  \  / \
    # #   \/   '.__
    # # Since the shape of the <Iqtau>tau is as above, looking for the negative
    # # peak and then mediating over q only from then on

    # itau = np.mean(iqtau, axis=2)
    
    # qstart = 0
    
    # q_temp = 0
    # for idx in range(len(q_vector) - 1):
    #     if q_vector[idx] > q_vector[idx+1]:
    #         q_start = idx
    #         break
    
    # if q_start == 0 or idx > len(q_vector) / 2:
    #     q_start = 0
    
    # DONT BOTHER preliminary fit of the <Iqtau>q to get good parameters for the fit of each I(q',tau)

    parameters = []

    for q_idx in range(len(q_vector)):
        xx = tau_vector
        yy = iqtau[q_idx]

        print(f"[{q_idx}]\t q  : {q_vector[q_idx]}")
        parameters.append(opt.curve_fit(dampend_osc, xx, yy)[0])

    return parameters

if __name__ == '__main__':
    q_vector, tau_vector, iqt = read_file("/home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt")
    out = fit(iqt, q_vector, tau_vector)


    fig, ax = plt.subplots()
    ax.set(xlabel=r"Lag time $\tau$", ylabel=r"I(q, $\tau$)")


    for idx, q in enumerate(q_vector):
        if idx % 3 != 0:
            continue

        fitted = np.array([dampend_osc(t, out[idx][0], out[idx][1], out[idx][2], out[idx][3]) for t in tau_vector])

        ax.plot(tau_vector, iqt[idx], label=f"q = {q}")
        ax.plot(tau_vector, fitted, label=f"fiited q = {q}")

ax.legend(loc="upper left")
plt.show()
