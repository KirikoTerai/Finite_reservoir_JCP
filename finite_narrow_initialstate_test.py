import numpy as np
import math
import matplotlib.pyplot as plt
from finite_functions import *
import sys

# # Redirect stdout to save printed statements to a file
# output_file = "run_finite.txt"
# original_stdout = sys.stdout
# sys.stdout = open(output_file, "w")

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

    # Calculate the determinant
    determinant = np.linalg.det(net.K)
    print("determinant=", determinant)

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
time = np.arange(0, 300+dt, dt)
dflux_time = []
## 010 electrons ##
pop_A_t_010 = []
pop_B_t_010 = []
pop_Left_t_010 = []
pop_Right_t_010 = []
flux_left_t_010 = []
flux_right_t_010 = []
## 28 electrons ##
pop_A_t_28 = []
pop_B_t_28 = []
pop_Left_t_28 = []
pop_Right_t_28 = []
flux_left_t_28 = []
flux_right_t_28 = []
## 46 electrons ##
pop_A_t_46 = []
pop_B_t_46 = []
pop_Left_t_46 = []
pop_Right_t_46 = []
flux_left_t_46 = []
flux_right_t_46 = []
## 55 electrons ##
pop_A_t_55 = []
pop_B_t_55 = []
pop_Left_t_55 = []
pop_Right_t_55 = []
flux_left_t_55 = []
flux_right_t_55 = []
## 64 electrons ##
pop_A_t_64 = []
pop_B_t_64 = []
pop_Left_t_64 = []
pop_Right_t_64 = []
flux_left_t_64 = []
flux_right_t_64 = []
## 73 electrons ##
pop_A_t_73 = []
pop_B_t_73 = []
pop_Left_t_73 = []
pop_Right_t_73 = []
flux_left_t_73 = []
flux_right_t_73 = []
## 82 electrons ##
pop_A_t_82 = []
pop_B_t_82 = []
pop_Left_t_82 = []
pop_Right_t_82 = []
flux_left_t_82 = []
flux_right_t_82 = []
## 100 electrons ##
pop_A_t_100 = []
pop_B_t_100 = []
pop_Left_t_100 = []
pop_Right_t_100 = []
flux_left_t_100 = []
flux_right_t_100 = []

for t in time:
    ## 010 ##
    pop_A_010, pop_B_010, flux_left_010, flux_right_010, pop_MEK_010, pop_Left_010, pop_Right_010 = run_finite(t, 0, 10)
    pop_A_t_010.append(pop_A_010)
    pop_B_t_010.append(pop_B_010)
    pop_Left_t_010.append(pop_Left_010)
    pop_Right_t_010.append(pop_Right_010)
    flux_left_t_010.append(flux_left_010)
    flux_right_t_010.append(flux_right_010)
    ## 28 ##
    pop_A_28, pop_B_28, flux_left_28, flux_right_28, pop_MEK_28, pop_Left_28, pop_Right_28 = run_finite(t, 2, 8)
    pop_A_t_28.append(pop_A_28)
    pop_B_t_28.append(pop_B_28)
    pop_Left_t_28.append(pop_Left_28)
    pop_Right_t_28.append(pop_Right_28)
    flux_left_t_28.append(flux_left_28)
    flux_right_t_28.append(flux_right_28)
    ## 46 ##
    pop_A_46, pop_B_46, flux_left_46, flux_right_46, pop_MEK_46, pop_Left_46, pop_Right_46 = run_finite(t, 4, 6)
    pop_A_t_46.append(pop_A_46)
    pop_B_t_46.append(pop_B_46)
    pop_Left_t_46.append(pop_Left_46)
    pop_Right_t_46.append(pop_Right_46)
    flux_left_t_46.append(flux_left_46)
    flux_right_t_46.append(flux_right_46)
    ## 55 ##
    pop_A_55, pop_B_55, flux_left_55, flux_right_55, pop_MEK_55, pop_Left_55, pop_Right_55 = run_finite(t, 5, 5)
    pop_A_t_55.append(pop_A_55)
    pop_B_t_55.append(pop_B_55)
    pop_Left_t_55.append(pop_Left_55)
    pop_Right_t_55.append(pop_Right_55)
    flux_left_t_55.append(flux_left_55)
    flux_right_t_55.append(flux_right_55)
    ## 64 ##
    pop_A_64, pop_B_64, flux_left_64, flux_right_64, pop_MEK_64, pop_Left_64, pop_Right_64 = run_finite(t, 6, 4)
    pop_A_t_64.append(pop_A_64)
    pop_B_t_64.append(pop_B_64)
    pop_Left_t_64.append(pop_Left_64)
    pop_Right_t_64.append(pop_Right_64)
    flux_left_t_64.append(flux_left_64)
    flux_right_t_64.append(flux_right_64)
    ## 82 ##
    pop_A_82, pop_B_82, flux_left_82, flux_right_82, pop_MEK_82, pop_Left_82, pop_Right_82 = run_finite(t, 8, 2)
    pop_A_t_82.append(pop_A_82)
    pop_B_t_82.append(pop_B_82)
    pop_Left_t_82.append(pop_Left_82)
    pop_Right_t_82.append(pop_Right_82)
    flux_left_t_82.append(flux_left_82)
    flux_right_t_82.append(flux_right_82)
    ## 100 ##
    pop_A_100, pop_B_100, flux_left_100, flux_right_100, pop_MEK_100, pop_Left_100, pop_Right_100 = run_finite(t, 10, 0)
    pop_A_t_100.append(pop_A_100)
    pop_B_t_100.append(pop_B_100)
    pop_Left_t_100.append(pop_Left_100)
    pop_Right_t_100.append(pop_Right_100)
    flux_left_t_100.append(flux_left_100)
    flux_right_t_100.append(flux_right_100)

plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')

## Populations ##
fig010 = plt.figure()
plt.plot(time, pop_A_t_010, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_010, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig010.savefig("finite_pop_010.pdf")

fig28 = plt.figure()
plt.plot(time, pop_A_t_28, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_28, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig28.savefig("finite_pop_28.pdf")

fig46 = plt.figure()
plt.plot(time, pop_A_t_46, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_46, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig46.savefig("finite_pop_46.pdf")

fig55 = plt.figure()
plt.plot(time, pop_A_t_55, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_55, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig55.savefig("finite_pop_55.pdf")

fig64 = plt.figure()
plt.plot(time, pop_A_t_64, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_64, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig64.savefig("finite_pop_64.pdf")

fig82 = plt.figure()
plt.plot(time, pop_A_t_82, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_82, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig82.savefig("finite_pop_82.pdf")

fig100 = plt.figure()
plt.plot(time, pop_A_t_100, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_100, label="p$_{B}$", color="blue")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
#plt.legend()
plt.tight_layout()
# plt.show()
fig100.savefig("finite_pop_100.pdf")


#### Flux #####
fig010f = plt.figure()
plt.plot(time, flux_left_t_010, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_010, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig010f.savefig("finite_flux_010.pdf")

fig28f = plt.figure()
plt.plot(time, flux_left_t_28, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_28, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig28f.savefig("finite_flux_28.pdf")

fig46f = plt.figure()
plt.plot(time, flux_left_t_46, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_46, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig46f.savefig("finite_flux_46.pdf")

fig55f = plt.figure()
plt.plot(time, flux_left_t_55, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_55, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig55f.savefig("finite_flux_55.pdf")

fig64f = plt.figure()
plt.plot(time, flux_left_t_64, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_64, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig64f.savefig("finite_flux_64.pdf")

fig82f = plt.figure()
plt.plot(time, flux_left_t_82, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_82, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig82f.savefig("finite_flux_82.pdf")

fig100f = plt.figure()
plt.plot(time, flux_left_t_100, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_100, label="Flux into Right reservoir", color="green")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
#plt.legend()
plt.tight_layout()
fig100f.savefig("finite_flux_100.pdf")
