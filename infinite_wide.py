import numpy as np
import math
import matplotlib.pyplot as plt
from infinite_wide_functions import *
import sys

# # Redirect stdout to save printed statements to a file
# output_file = "run_Nitzan.txt"
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

def run_Nitzan(t, mu_L, mu_R, initial_state:list):
    net = Network()

    # e_A = e_A_o + alpha*(1+np.sin(omega*t)) + F/3
    # e_B = e_B_o + alpha*(1+np.sin(omega*t-math.pi/2)) +2*F/3
    # e_A = e_A_o + 2*alpha*(1+np.sin(omega*t)) + F/3
    # e_B = e_B_o + alpha*(1+np.sin(omega*t)) +2*F/3
    e_A = e_A_o
    e_B = e_B_o

    A = Site("A", e_A) 
    B = Site("B", e_B)

    net.addSite(A)
    net.addSite(B)

    net.addReservoir("Left", e_L, mu_L)
    net.addReservoir("Right", e_R, mu_R)

    net.initializeRateMatrix()

    net.connectSite(A, B, 10**(0))
    net.connectReservoir(A, "Left", 10**(0))
    net.connectReservoir(B, "Right", 10**(0))

    # K_eval, K_evec = np.linalg.eig(net.K)
    # print(K_evec)

    print(net.K)

    # Calculate the determinant
    determinant = np.linalg.det(net.K)
    print("determinant=", determinant)

    state_idx = net.state2idx(initial_state)
    pop_MEK_init = np.zeros(net.num_state)
    pop_MEK_init[state_idx] = 1

    pop_MEK = net.evolve(t, pop_MEK_init)

    print(t, pop_MEK[0], pop_MEK[1], pop_MEK[2], pop_MEK[3])

    pop_A = net.population(pop_MEK, A, 1)
    pop_B = net.population(pop_MEK, B, 1)
    pop_L = round(net.reservoir_mean_occupation("Left"), 4)
    pop_R = round(net.reservoir_mean_occupation("Right"), 4)

    def get_LeftFlux(pop: np.array):
        flux = 0
        # Forward: site A -> left reservoir
        site = A
        site_id = net.site2id[site]
        for initial in range(net.num_state):
            for final in range(net.num_state):
                if net.idx2state(initial)[site_id] == 1 and net.idx2state(final)[site_id] == 0:
                    # initial, final state found! check other electron conservation
                    I = np.delete(net.idx2state(initial), [site_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                    J = np.delete(net.idx2state(final), [site_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                    if np.array_equal(I, J):
                        kf = net.K[final][initial]
                        kb = net.K[initial][final]
                        # print(net.idx2state(initial), net.idx2state(final), kf, kb)
                        flux += pop[initial] * kf
                        flux -= pop[final] * kb
        return flux
    
    def get_RightFlux(pop:np.array):
        flux = 0
        # Forward: site B -> right reservoir
        site = B
        site_id = net.site2id[site]
        for initial in range(net.num_state):
            for final in range(net.num_state):
                if net.idx2state(initial)[site_id] == 1 and net.idx2state(final)[site_id] == 0:
                    # initial, final state found! check other electron conservation
                    I = np.delete(net.idx2state(initial), [site_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                    J = np.delete(net.idx2state(final), [site_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                    if np.array_equal(I, J):
                        kf = net.K[final][initial]
                        kb = net.K[initial][final]
                        # print(net.idx2state(initial), net.idx2state(final), kf, kb)
                        flux += pop[initial] * kf
                        flux -= pop[final] * kb
        return flux

    flux_left = get_LeftFlux(pop_MEK)
    flux_right = get_RightFlux(pop_MEK)

    # print("Population", pop_A, pop_B)
    # print("Reservoir density", pop_L, pop_R)
    # print("Flux", flux_left, flux_right)
    print("-------------------------------------")

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(net.K)

    # print(eigenvalues)
    # print(eigenvectors)

    # Filter out zero eigenvalues (or very close to zero, considering numerical precision)
    nonzero_eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-10]
    # Get the eigenvalue that is equal to zero
    zero_eigenvalue = eigenvalues[np.abs(eigenvalues) < 1e-12]

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
    zero_index = np.where(eigenvalues == zero_eigenvalue)[0][0]

    # Extract the corresponding eigenvectors (eigenstates)
    dominant_eigenvector = eigenvectors[:, dominant_index]
    second_dominant_eigenvector = eigenvectors[:, second_dominant_index]
    third_dominant_eigenvector = eigenvectors[:, third_dominant_index]
    zero_eigenvector = eigenvectors[:, zero_index]

    # Normalize the eigenvectors for interpretability
    dominant_eigenvector /= np.linalg.norm(dominant_eigenvector)
    second_dominant_eigenvector /= np.linalg.norm(second_dominant_eigenvector)
    third_dominant_eigenvector /= np.linalg.norm(third_dominant_eigenvector)
    zero_eigenvector = zero_eigenvector/np.sum(zero_eigenvector)

    print("Steady-eigenstate:", zero_eigenvector)

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

    return pop_A, pop_B, flux_left, flux_right, pop_MEK, e_A, e_B, pop_L, pop_R

initial_state = [0, 0]
mu_L_dense = 0
mu_R_dense = 0
mu_L_moderate = -2.5
mu_R_moderate = -2.5
mu_L_dilute = -5
mu_R_dilute = -5

dt = 0.01
time = np.arange(0, 50+dt, dt)
## Dilute limit ##
pop_A_t_dilute = []
pop_B_t_dilute = []
pop_L_t_dilute = []
pop_R_t_dilute = []
flux_left_t_dilute = []
flux_right_t_dilute = []
pop_00_t_dilute = []
pop_10_t_dilute = []
pop_01_t_dilute = []
pop_11_t_dilute = []
mu_L_dilute = -5
mu_R_dilute = -5
## Dense system ##
pop_A_t_dense = []
pop_B_t_dense = []
pop_L_t_dense = []
pop_R_t_dense = []
flux_left_t_dense = []
flux_right_t_dense = []
pop_00_t_dense = []
pop_10_t_dense = []
pop_01_t_dense = []
pop_11_t_dense = []
mu_L_dense = 0
mu_R_dense = 0
## moderate system ##
pop_A_t_moderate = []
pop_B_t_moderate = []
pop_L_t_moderate = []
pop_R_t_moderate = []
flux_left_t_moderate = []
flux_right_t_moderate = []
pop_00_t_moderate = []
pop_10_t_moderate = []
pop_01_t_moderate = []
pop_11_t_moderate = []
mu_L_moderate = -2.5
mu_R_moderate = -2.5
for t in time:
    ## Dilute limit ##
    pop_A_dilute, pop_B_dilute, flux_left_dilute, flux_right_dilute, pop_MEK_dilute, e_A_dilute, e_B_dilute, pop_L_dilute, pop_R_dilute = run_Nitzan(t, mu_L_dilute, mu_R_dilute, initial_state)
    pop_A_t_dilute.append(pop_A_dilute)
    pop_B_t_dilute.append(pop_B_dilute)
    flux_left_t_dilute.append(flux_left_dilute)
    flux_right_t_dilute.append(flux_right_dilute)
    pop_00_t_dilute.append(pop_MEK_dilute[0])
    pop_10_t_dilute.append(pop_MEK_dilute[1])
    pop_01_t_dilute.append(pop_MEK_dilute[2])
    pop_11_t_dilute.append(pop_MEK_dilute[3])
    pop_L_t_dilute.append(pop_L_dilute)
    pop_R_t_dilute.append(pop_R_dilute)
    ## Dense system ##
    pop_A_dense, pop_B_dense, flux_left_dense, flux_right_dense, pop_MEK_dense, e_A_dense, e_B_dense, pop_L_dense, pop_R_dense = run_Nitzan(t, mu_L_dense, mu_R_dense, initial_state)
    pop_A_t_dense.append(pop_A_dense)
    pop_B_t_dense.append(pop_B_dense)
    flux_left_t_dense.append(flux_left_dense)
    flux_right_t_dense.append(flux_right_dense)
    pop_00_t_dense.append(pop_MEK_dense[0])
    pop_10_t_dense.append(pop_MEK_dense[1])
    pop_01_t_dense.append(pop_MEK_dense[2])
    pop_11_t_dense.append(pop_MEK_dense[3])
    pop_L_t_dense.append(pop_L_dense)
    pop_R_t_dense.append(pop_R_dense)
    ## moderate system ##
    pop_A_moderate, pop_B_moderate, flux_left_moderate, flux_right_moderate, pop_MEK_moderate, e_A_moderate, e_B_moderate, pop_L_moderate, pop_R_moderate = run_Nitzan(t, mu_L_moderate, mu_R_moderate, initial_state)
    pop_A_t_moderate.append(pop_A_moderate)
    pop_B_t_moderate.append(pop_B_moderate)
    flux_left_t_moderate.append(flux_left_moderate)
    flux_right_t_moderate.append(flux_right_moderate)
    pop_00_t_moderate.append(pop_MEK_moderate[0])
    pop_10_t_moderate.append(pop_MEK_moderate[1])
    pop_01_t_moderate.append(pop_MEK_moderate[2])
    pop_11_t_moderate.append(pop_MEK_moderate[3])
    pop_L_t_moderate.append(pop_L_moderate)
    pop_R_t_moderate.append(pop_R_moderate)

plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')

#### Expected number of electrons plot #####
fig1 = plt.figure()
plt.plot(time, pop_A_t_dense, label="p$_{A}$", color="black")
plt.plot(time, pop_B_t_dense, label="p$_{B}$", color="blue")
plt.plot(time, pop_A_t_moderate, color="black", linestyle="dashed")
plt.plot(time, pop_B_t_moderate, color="blue", linestyle="dashed")
plt.plot(time, pop_A_t_dilute, color="black", linestyle="dotted")
plt.plot(time, pop_B_t_dilute, color="blue", linestyle="dotted")
plt.title("Expected number of particles", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
# plt.legend(loc="lower right")
plt.ylim((-0.05, 1.05))
plt.tight_layout()
plt.show()
fig1.savefig("infinite_wide_pop.pdf")

#### Flux #####
fig2 = plt.figure()
plt.plot(time, flux_left_t_dense, label="Flux into Left reservoir", color="red")
plt.plot(time, flux_right_t_dense, label="Flux into Right reservoir", color="green")
plt.plot(time, flux_left_t_moderate, color="red", linestyle="dashed")
plt.plot(time, flux_right_t_moderate, color="green", linestyle="dashed")
plt.plot(time, flux_left_t_dilute, color="red", linestyle="dotted")
plt.plot(time, flux_right_t_dilute, color="green", linestyle="dotted")
plt.title("Flux (s$^{-1}$)", size='xx-large')
plt.xlabel("Time (s)", size='xx-large')
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
# plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
fig2.savefig("infinite_wide_flux.pdf")
