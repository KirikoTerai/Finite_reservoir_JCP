import numpy as np
import math
import matplotlib.pyplot as plt
from finite_functions import *
import sys

# Constants  (energy units: kBT)
e_A_o = -2
e_B_o = -2
tao = 2
omega = 2*math.pi/tao
F = 1.5
e_L = 0     # Energy of left reservoir
e_R = F     # Energy of right reservoir
alpha = 5

ss_time = []

def run_finite(t, left_electron, right_electron):
    num_electron = left_electron + right_electron

    # mu_L and mu_R are not used in finite model. The values can be anything
    net = Network()

    # e_A = e_A_o + alpha*(1+np.sin(omega*t)) + F/3
    # e_B = e_B_o + alpha*(1+np.sin(omega*t-math.pi/2)) +2*F/3
    # e_A = e_A_o + 2*alpha*(1+np.sin(omega*t)) + F/3
    # e_B = e_B_o + alpha*(1+np.sin(omega*t)) +2*F/3
    e_A = e_A_o
    e_B = e_B_o

    Left = Site("Left", e_L, num_electron)
    A = Site("A", e_A, 1) 
    B = Site("B", e_B, 1)
    Right = Site("Right", e_R, num_electron)

    net.addSite(Left)
    net.addSite(A)
    net.addSite(B)
    net.addSite(Right)

    net.set_Max_Electrons(num_electron)
    net.set_Min_Electrons(num_electron)
    net.constructStateList()

    net.initializeRateMatrix()

    def makeRateMatrix():
        res_left_id = net.site2id[Left]
        site_A_id = net.site2id[A]
        site_B_id = net.site2id[B]
        res_right_id = net.site2id[Right]
        rate = 1
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if initial != final:
                    # Site A -> site B
                    if net.idx2state(net.allow[initial])[site_A_id]-net.idx2state(net.allow[final])[site_A_id]==1:
                        if net.idx2state(net.allow[final])[site_B_id]-net.idx2state(net.allow[initial])[site_B_id]==1:
                            # initial, final state found! check other electron conservation
                            I = np.delete(net.idx2state(net.allow[initial]), [site_A_id, site_B_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                            J = np.delete(net.idx2state(net.allow[final]), [site_A_id, site_B_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                            if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                                deltaG = 0
                                kf = rate*np.exp(net.beta*deltaG/2)
                                kb = rate*np.exp(-net.beta*deltaG/2)
                                net.K[final][initial] += kf
                                net.K[initial][initial] -= kf
                                net.K[initial][final] += kb
                                net.K[final][final] -= kb
                    # Site A -> left resevoir
                    if net.idx2state(net.allow[initial])[site_A_id]-net.idx2state(net.allow[final])[site_A_id]==1:
                        if net.idx2state(net.allow[final])[res_left_id]-net.idx2state(net.allow[initial])[res_left_id]==1:
                            # initial, final state found! check other electron conservation
                            I = np.delete(net.idx2state(net.allow[initial]), [site_A_id, res_left_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                            J = np.delete(net.idx2state(net.allow[final]), [site_A_id, res_left_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                            if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                                deltaG = e_A-e_L
                                kf = rate*np.exp(net.beta*deltaG/2)
                                kb = rate*np.exp(-net.beta*deltaG/2)
                                net.K[final][initial] += kf
                                net.K[initial][initial] -= kf
                                net.K[initial][final] += kb
                                net.K[final][final] -= kb
                    # Site B -> right resevoir
                    if net.idx2state(net.allow[initial])[site_B_id]-net.idx2state(net.allow[final])[site_B_id]==1:
                        if net.idx2state(net.allow[final])[res_right_id]-net.idx2state(net.allow[initial])[res_right_id]==1:
                            # initial, final state found! check other electron conservation
                            I = np.delete(net.idx2state(net.allow[initial]), [site_B_id, res_right_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                            J = np.delete(net.idx2state(net.allow[final]), [site_B_id, res_right_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                            if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                                deltaG = e_B-e_R
                                kf = rate*np.exp(net.beta*deltaG/2)
                                kb = rate*np.exp(-net.beta*deltaG/2)
                                net.K[final][initial] += kf
                                net.K[initial][initial] -= kf
                                net.K[initial][final] += kb
                                net.K[final][final] -= kb

    makeRateMatrix()

    initial_state = [left_electron, 0, 0, right_electron]
    pop_MEK_init = np.zeros(net.adj_num_state)
    for allow_idx in net.allow:
        if net.state2idx(initial_state) == allow_idx:
            pop_MEK_init_idx = net.allow.index(allow_idx)
    pop_MEK_init[pop_MEK_init_idx] = 1

    pop_MEK = net.evolve(t, pop_MEK_init)

    pop_A = net.population(pop_MEK, A, 1)
    pop_B = net.population(pop_MEK, B, 1)
    pop_Left = net.getExptvalue(pop_MEK, Left)
    pop_Right = net.getExptvalue(pop_MEK, Right)

    pop_MEK1 = net.evolve(t+dt, pop_MEK_init)

    pop_A1 = net.population(pop_MEK1, A, 1)
    pop_B1 = net.population(pop_MEK1, B, 1)

    dpop_A = (pop_A1-pop_A)/dt
    dpop_B = (pop_B1-pop_B)/dt

    def get_LeftFlux(pop: np.array):
        flux = 0
        # Forward: site A -> left reservoir
        site = A
        res = Left
        site_id = net.site2id[site]
        res_id = net.site2id[res]
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if net.idx2state(net.allow[initial])[site_id]-net.idx2state(net.allow[final])[site_id]==1:
                    if net.idx2state(net.allow[final])[res_id]-net.idx2state(net.allow[initial])[res_id]==1:
                        # initial, final state found! check other electron conservation
                        I = np.delete(net.idx2state(net.allow[initial]), [site_id, res_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(net.idx2state(net.allow[final]), [site_id, res_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):
                            kf = net.K[final][initial]
                            kb = net.K[initial][final]
                            # print(net.idx2state[initial], net.idx2state[final], kf, kb)
                            flux += pop[initial] * kf
                            flux -= pop[final] * kb
        return flux
    
    def get_RightFlux(pop: np.array):
        flux = 0
        # Forward: site B -> right reservoir
        site = B
        res = Right
        site_id = net.site2id[site]
        res_id = net.site2id[res]
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if net.idx2state(net.allow[initial])[site_id]-net.idx2state(net.allow[final])[site_id]==1:
                    if net.idx2state(net.allow[final])[res_id]-net.idx2state(net.allow[initial])[res_id]==1:
                        # initial, final state found! check other electron conservation
                        I = np.delete(net.idx2state(net.allow[initial]), [site_id, res_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(net.idx2state(net.allow[final]), [site_id, res_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):
                            kf = net.K[final][initial]
                            kb = net.K[initial][final]
                            # print(net.idx2state[initial], net.idx2state[final], kf, kb)
                            flux += pop[initial] * kf
                            flux -= pop[final] * kb
        return flux
    
    flux_left = get_LeftFlux(pop_MEK)
    flux_right = get_RightFlux(pop_MEK)

    print("----------------","time=",t,"-------------------")

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(net.K)

    # Filter out zero eigenvalues (or very close to zero, considering numerical precision)
    nonzero_eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]

    # Sort eigenvalues by magnitude (absolute value) to determine dominant modes
    sorted_indices = np.argsort(np.abs(nonzero_eigenvalues))

    # Identify the most, second most, and third most dominant eigenvalues
    dominant_eigenvalue = nonzero_eigenvalues[sorted_indices[0]]
    second_dominant_eigenvalue = nonzero_eigenvalues[sorted_indices[1]]
    third_dominant_eigenvalue = nonzero_eigenvalues[sorted_indices[2]]

    # Find indices of the corresponding eigenvalues in the original list
    dominant_index = np.where(eigenvalues == dominant_eigenvalue)[0][0]
    second_dominant_index = np.where(eigenvalues == second_dominant_eigenvalue)[0][0]
    third_dominant_index = np.where(eigenvalues == third_dominant_eigenvalue)[0][0]

    # Extract the corresponding eigenvectors (eigenstates)
    dominant_eigenvector = eigenvectors[:, dominant_index]
    second_dominant_eigenvector = eigenvectors[:, second_dominant_index]
    third_dominant_eigenvector = eigenvectors[:, third_dominant_index]

    # Normalize the eigenvectors for interpretability
    dominant_eigenvector /= np.linalg.norm(dominant_eigenvector)
    second_dominant_eigenvector /= np.linalg.norm(second_dominant_eigenvector)
    third_dominant_eigenvector /= np.linalg.norm(third_dominant_eigenvector)

    # Compute projections of the initial state onto the eigenvectors
    projection_1 = abs(np.dot(dominant_eigenvector, pop_MEK_init))
    projection_2 = abs(np.dot(second_dominant_eigenvector, pop_MEK_init))
    projection_3 = abs(np.dot(third_dominant_eigenvector, pop_MEK_init))

    # Print the results
    print("Eigenvalue determining the most dominant time scale:", dominant_eigenvalue)
    print("Timescale of most dominant mode:", -1 / dominant_eigenvalue)
    print("Projection onto most dominant mode:", projection_1)
    print("-----------")
    print("Eigenvalue determining the second most dominant time scale:", second_dominant_eigenvalue)
    print("Timescale of second most dominant mode:", -1 / second_dominant_eigenvalue)
    print("Projection onto second most dominant mode:", projection_2)
    print("-----------")
    print("Eigenvalue determining the third most dominant time scale:", third_dominant_eigenvalue)
    print("Timescale of third most dominant mode:", -1 / third_dominant_eigenvalue)
    print("Projection onto third most dominant mode:", projection_3)

    return pop_A, pop_B, flux_left, flux_right, pop_MEK, pop_Left, pop_Right

dt = 0.1
time = np.arange(0, 500+dt, dt)
## 2 electrons ##
pop_A_t_2 = []
pop_B_t_2 = []
pop_Left_t_2 = []
pop_Right_t_2 = []
flux_left_t_2 = []
flux_right_t_2 = []
## 4 electrons ##
pop_A_t_4 = []
pop_B_t_4 = []
pop_Left_t_4 = []
pop_Right_t_4 = []
flux_left_t_4 = []
flux_right_t_4 = []
## 6 electrons ##
pop_A_t_6 = []
pop_B_t_6 = []
pop_Left_t_6 = []
pop_Right_t_6 = []
flux_left_t_6 = []
flux_right_t_6 = []
## 8 electrons ##
pop_A_t_8 = []
pop_B_t_8 = []
pop_Left_t_8 = []
pop_Right_t_8 = []
flux_left_t_8 = []
flux_right_t_8 = []
## 10 electrons ##
pop_A_t_10 = []
pop_B_t_10 = []
pop_Left_t_10 = []
pop_Right_t_10 = []
flux_left_t_10 = []
flux_right_t_10 = []
## 12 electrons ##
pop_A_t_12 = []
pop_B_t_12 = []
pop_Left_t_12 = []
pop_Right_t_12 = []
flux_left_t_12 = []
flux_right_t_12 = []

for t in time:
    ## 2 electrons ##
    pop_A_2, pop_B_2, flux_left_2, flux_right_2, pop_MEK_2, pop_Left_2, pop_Right_2 = run_finite(t, 0, 2)
    pop_A_t_2.append(pop_A_2)
    pop_B_t_2.append(pop_B_2)
    pop_Left_t_2.append(pop_Left_2)
    pop_Right_t_2.append(pop_Right_2)
    flux_left_t_2.append(flux_left_2)
    flux_right_t_2.append(flux_right_2)
    ## 4 electrons ##
    pop_A_4, pop_B_4, flux_left_4, flux_right_4, pop_MEK_4, pop_Left_4, pop_Right_4 = run_finite(t, 0, 4)
    pop_A_t_4.append(pop_A_4)
    pop_B_t_4.append(pop_B_4)
    pop_Left_t_4.append(pop_Left_4)
    pop_Right_t_4.append(pop_Right_4)
    flux_left_t_4.append(flux_left_4)
    flux_right_t_4.append(flux_right_4)
    ## 6 electrons ##
    pop_A_6, pop_B_6, flux_left_6, flux_right_6, pop_MEK_6, pop_Left_6, pop_Right_6 = run_finite(t, 0, 6)
    pop_A_t_6.append(pop_A_6)
    pop_B_t_6.append(pop_B_6)
    pop_Left_t_6.append(pop_Left_6)
    pop_Right_t_6.append(pop_Right_6)
    flux_left_t_6.append(flux_left_6)
    flux_right_t_6.append(flux_right_6)
    ## 8 electrons ##
    pop_A_8, pop_B_8, flux_left_8, flux_right_8, pop_MEK_8, pop_Left_8, pop_Right_8 = run_finite(t, 0, 8)
    pop_A_t_8.append(pop_A_8)
    pop_B_t_8.append(pop_B_8)
    pop_Left_t_8.append(pop_Left_8)
    pop_Right_t_8.append(pop_Right_8)
    flux_left_t_8.append(flux_left_8)
    flux_right_t_8.append(flux_right_8)
    ## 10 electrons ##
    pop_A_10, pop_B_10, flux_left_10, flux_right_10, pop_MEK_10, pop_Left_10, pop_Right_10 = run_finite(t, 0, 10)
    pop_A_t_10.append(pop_A_10)
    pop_B_t_10.append(pop_B_10)
    pop_Left_t_10.append(pop_Left_10)
    pop_Right_t_10.append(pop_Right_10)
    flux_left_t_10.append(flux_left_10)
    flux_right_t_10.append(flux_right_10)
    ## 12 electrons ##
    pop_A_12, pop_B_12, flux_left_12, flux_right_12, pop_MEK_12, pop_Left_12, pop_Right_12 = run_finite(t, 0, 12)
    pop_A_t_12.append(pop_A_12)
    pop_B_t_12.append(pop_B_12)
    pop_Left_t_12.append(pop_Left_12)
    pop_Right_t_12.append(pop_Right_12)
    flux_left_t_12.append(flux_left_12)
    flux_right_t_12.append(flux_right_12)

plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')

#### Flux no ss #####
fig2 = plt.figure()
plt.plot(time, flux_left_t_2, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_2, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig2.savefig("finite_flux_2.pdf")

fig2p = plt.figure()
plt.plot(time, pop_A_t_2, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_2, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig2p.savefig("finite_pop_2.pdf")

fig4 = plt.figure()
plt.plot(time, flux_left_t_4, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_4, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig4.savefig("finite_flux_4.pdf")

fig4p = plt.figure()
plt.plot(time, pop_A_t_4, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_4, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig4p.savefig("finite_pop_4.pdf")


#### Flux ss #####
fig6 = plt.figure()
plt.plot(time, flux_left_t_6, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_6, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig6.savefig("finite_flux_6.pdf")

fig6p = plt.figure()
plt.plot(time, pop_A_t_6, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_6, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig6p.savefig("finite_pop_6.pdf")


fig8 = plt.figure()
plt.plot(time, flux_left_t_8, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_8, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig8.savefig("finite_flux_8.pdf")

fig8p = plt.figure()
plt.plot(time, pop_A_t_8, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_8, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig8p.savefig("finite_pop_8.pdf")

fig10 = plt.figure()
plt.plot(time, flux_left_t_10, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_10, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig10.savefig("finite_flux_10.pdf")

fig10p = plt.figure()
plt.plot(time, pop_A_t_10, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_10, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig10p.savefig("finite_pop_10.pdf")

fig12 = plt.figure()
plt.plot(time, flux_left_t_12, color="red", label="Flux into Left reservoir")
plt.plot(time, flux_right_t_12, color="green", label="Flux into Right reservoir")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
# plt.show()
fig12.savefig("finite_flux_12.pdf")

fig12p = plt.figure()
plt.plot(time, pop_A_t_12, color="black", label="p$_{A}$")
plt.plot(time, pop_B_t_12, color="blue", label="p$_{B}$")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig12p.savefig("finite_pop_12.pdf")
