import numpy as np
from scipy import linalg
from scipy.optimize import *

class Site():
    def __init__(self, name: str, energy: float):
        """
        Initialize this site object, with property: name, and site energies
        Arguments:
            name {str} -- Name of the site
            energy {list} -- List of energies of the site
        """
        self.name = name
        self.energy = energy          #(ex.) "[first reduction potential (0 -> 1), second reduction potential (1 -> 2),...]
        # !!!! Only assuming sites that can occupy up to one particle
        self.capacity = 1    # The number of electrons the site can occupy is equal to the number of reduction potentials

    def __str__(self) -> str:         #__str__:a built-in function that computes the "informal" string representations of an object
        """
        Return a string representation of the site
        Returns:
            str -- String representation of the site
        """
        s = ""
        # Initialize with site name
        s += "Site Name: {}\n".format(self.name)     #\n:new line in string
        s += "------------ \n"     #Draw a line between cofactor info (looks cuter!)
        # Print with state_id and relative site energy
        s += "Energy {}\n".format(self.energy)

        return s
    

class Network():
    def __init__(self):
        """
        Initialize the whole system
        NOTICE: the initialized Network instance has nothing in it, use other functions to insert information
        """
        # system-specific data structure and parameters
        self.num_site = 0
        self.num_state = 1
        self.id2site = dict()  # key-value mapping is id-cofactor
        self.site2id = dict()  # key-value mapping is cofactor-id
        self.adjacencyList = list()
        self.siteCapacity = []  # indexing is id-site_capacity
        self.D = None  # not defined
        self.K = None  # not defined
        self.num_reservoir = 0
        self.reservoirInfo = dict()    # key-value mapping is id-reservoir name, cofactor, redox_state, num_electron, deltaG, rate
        self.id2reservoir=dict()    # key-value mapping is id-reservoir name
        self.reservoir2id=dict()    # key-value mapping is reservoir name-id
        self.max_electrons = None
        """
        ET-specific data structure and parameters     #Incorporate to the ET function?
        """
        self.hbar = 6.5821 * 10 ** (-16)  # unit: eV sec
        # self.beta = 39.06  # unit: 1/kT in 1/eV (room temperature)
        self.beta = 1  # Units in kBT!!
        self.reorgE = 0.7 # unit: eV
        self.V = 0.1 # unit: eV

    def addSite(self, site: Site):
        """
        Add site into this Network
        Arguments:
            site {Site} -- Site object
        """
        self.num_state *= 2        # The total number of possible states is equal to the product of sitecapacities+1 of each site.
                                             # (ex.) "Cofactor_1":0,1, "Cofactor_2":0,1,2 -> num_states=(cap_1+1)*(cap_2+1)=(1+1)*(2+1)=2*3=6

        self.id2site[self.num_site] = site   #Starts with self.num_cofactor=0, Gives an ID to cofactors that are added one by one
        self.site2id[site] = self.num_site   #ID of the cofactor just added is basically equal to how many cofactors present in the network
        self.siteCapacity.append(site.capacity)    #Trajectory of cofactor -> id -> capacity of cofactor
        self.num_site += 1    #The number of cofactor counts up

    def addReservoir(self, name: str, energy: float, potential: float):
        """
        Add an electron reservoir to the network: which cofactor it exchanges electrons with, how many electrons are exchanged at a time, the deltaG of the exchange, and the rate
        Arguments:
            name {str} -- Name of the reservoir
            site {Site} -- Site the reservoir exchanges electrons
            energy {float} -- energy of the reservoir
        """
        # key: (reservoir_id, cofactor_id)
        # value: list of six variables, [name, cofactor, redox_state, num_electron, deltaG, rate]
        self.id2reservoir[self.num_reservoir] = name
        self.reservoir2id[name] = self.num_reservoir
        self.reservoirInfo[self.num_reservoir] = [name, energy, potential]
        self.num_reservoir += 1

    def state2idx(self, state: list) -> int:
        """
        Given the list representation of the state, return index number in the main rate matrix
        Arguments:
            state {list} -- List representation of the state
        Returns:
            int -- Index number of the state in the main rate matrix
        """
        idx = 0
        N = 1
        for i in range(self.num_site):
            idx += state[i] * N
            N *= (self.siteCapacity[i] + 1)

        return idx

    def idx2state(self, idx: int) -> list:
        """
        Given the index number of the state in the main rate matrix, return the list representation of the state
        Arguments:
            idx {int} -- Index number of the state in the main rate matrix
        Returns:
            list -- List representation of the state
        """
        state = []
        for i in range(self.num_site):
            div = self.siteCapacity[i] + 1
            idx, num = divmod(idx, div)
            state.append(num)

        return state
    
    def state_energy(self, state_idx: int):
        state = self.idx2state(state_idx)
        state_energy = 0
        for i in range(len(state)):
            site_occupancy = state[i]
            if site_occupancy == 1:
                site = self.id2site[i]
                state_energy += site.energy
            if site_occupancy == 0:
                state_energy += 0
        return state_energy
    
    def reservoir_mean_occupation(self, reservoir_name: str):
        reservoir_id = self.reservoir2id[reservoir_name]
        reservoir_info = self.reservoirInfo[reservoir_id]
        reservoir_potential = reservoir_info[2]
        return (np.exp(-self.beta*reservoir_potential)+1)**(-1)

    def initializeRateMatrix(self):
        # initialize the rate matrix with proper dimension
        self.K = np.zeros((self.num_state, self.num_state), dtype=float)      #The dimension of the rate matrix is basically equal to the total number of states

    def connectSite(self, site_1: Site, site_2: Site, rate: float):
        site_1_id = self.site2id[site_1]
        site_2_id = self.site2id[site_2]
        # Find microstate transitions that represent particle transfer from site_1 to site_2
        for initial in range(self.num_state):
            for final in range(self.num_state):
                if self.idx2state(initial)[site_1_id] == 1 and self.idx2state(initial)[site_2_id] == 0:
                    if self.idx2state(final)[site_1_id] == 0 and self.idx2state(final)[site_2_id] == 1:
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(initial), [site_1_id, site_2_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(self.idx2state(final), [site_1_id, site_2_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                            # initial and final state found!
                            deltaG = self.state_energy(initial) - self.state_energy(final)
                            kf = rate*np.exp(self.beta*deltaG/2)
                            kb = rate*np.exp(-self.beta*deltaG/2)
                            # print(self.idx2state(initial), initial, "->", self.idx2state(final), final)
                            # print("kf=", kf, "->", "kb", kb)
                            # print("deltaG=",deltaG)
                            self.K[final][initial] += kf  # add population of final state, forward process
                            self.K[initial][initial] -= kf  # remove population of initial state, forward process   #Diagonal elements are the negative sum of the other elements in the same column
                            self.K[initial][final] += kb  # add population of initial state, backward process
                            self.K[final][final] -= kb  # remove population of final sate, backward process
    
    def connectReservoir(self, site: Site, reservoir_name: str, rate: float):
        site_id = self.site2id[site]
        reservoir_id = self.reservoir2id[reservoir_name]
        reservoir_info = self.reservoirInfo[reservoir_id]
        reservoir_energy = reservoir_info[1]
        # Find microstate transitions that represent particle transfer from site to reservoir
        for initial in range(self.num_state):
            for final in range(self.num_state):
                if self.idx2state(initial)[site_id] == 1 and self.idx2state(final)[site_id] == 0:
                    # initial, final state found! check other electron conservation
                    I = np.delete(self.idx2state(initial), [site_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                    J = np.delete(self.idx2state(final), [site_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                    if np.array_equal(I, J):
                        # initial and final state found!
                        p_res = self.reservoir_mean_occupation(reservoir_name)
                        deltaG = site.energy-reservoir_energy
                        kf = rate*np.exp(self.beta*deltaG/2)*(1-p_res)
                        kb = rate*np.exp(-self.beta*deltaG/2)*p_res
                        # print(self.idx2state(initial), initial, "->", self.idx2state(final), final)
                        # print("e_site=", site.energy, "e_res=", reservoir_energy, "deltaG=",deltaG, "p_res=", p_res)
                        # print("kf=", kf, "->", "kb", kb)
                        self.K[final][initial] += kf  # add population of final state, forward process
                        self.K[initial][initial] -= kf  # remove population of initial state, forward process   #Diagonal elements are the negative sum of the other elements in the same column
                        self.K[initial][final] += kb  # add population of initial state, backward process
                        self.K[final][final] -= kb  # remove population of final sate, backward process

    def evolve(self, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return linalg.expm(self.K * t) @ pop_init
    
    def rateEqn(self, state_idx: int, pop: np.array):
        ode = 0
        # pop: pop_MEK, population 
        for final_state_idx in range(self.num_state):
            # Find a state where self.idx2state(state_idx) is connected with
            if final_state_idx != state_idx:   # Don't get the diagonals
                if self.K[final_state_idx][state_idx] != 0:
                    # state found!
                    # print(self.idx2state(state_idx), "->", self.idx2state(final_state_idx))
                    # probability of the state represented by state_idx
                    p_initial = pop[state_idx]
                    p_final = pop[final_state_idx]
                    kf = self.K[final_state_idx][state_idx]
                    kb = self.K[state_idx][final_state_idx]
                    # print("kf=", kf, "kb=", kb)
                    ode -= kf*p_initial
                    ode += kb*p_final
        return ode
    
    def getProbabilityVector(self):
        num_unknowns = self.num_state
        init_prob_list = []
        N = 0
        while N < num_unknowns:
            N += 1
            q = np.random.rand()
            #init_prob_list.append(0)
            init_prob_list.append(q)
        # print(init_prob_list)
        return init_prob_list
    
    def getpop(self, RateEqns, pop_init):
        # Numerically solve set of non-linear equations
        # Core part of the mean-field model calculation!!
        pop = []    # pop = [P_00, P_10, P_01, P_11]
        success = False
        while success == False:
            try:
                pop = fsolve(RateEqns, pop_init)
                if all(0 < element < 1 for element in pop) and abs(1-sum(pop))<10**(-4):
                    success = True
                    # print('My initial guess succeeded!')
                else:
                    success = False
                    # print('The solutions given by your very first initial guess were unphysical. Try again...')
                    success2 = False
                    while success2 == False:
                        try:
                            probability_init = self.getProbabilityVector()       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                            # print('Generated new intial guess:', probability_init)
                            pop = fsolve(RateEqns, probability_init)
                            if all(0 < element < 1 for element in pop) and abs(1-sum(pop))<10**(-4):
                                success2 = True    # Terminate the while success2 == False loop
                                success = True     # Terminate the while success == False loop
                                # print('Successfully found physical solutions!')
                            else:
                                success2 = False
                                # print('The solutions were unphysical again. Try another initial guess...')
                        except ValueError:
                            success2 = False
                                # print('Oops! Have to try another initial guess...')
            except ValueError:
                # print('Oops! Could not find root within given tolerance using given initial guess...')
                success3 = False
                while success3 == False:
                    try:
                        probability_init = self.getProbabilityVector()       # Initial population of [p_ox, p_sq, p_L1, p_H1, p_L2, p_H2]
                        # print('Generated new intial guess:', probability_init)
                        pop = fsolve(RateEqns, probability_init)
                        if all(0 < element < 1 for element in pop) and abs(1-sum(pop))<10**(-4):
                            success3 = True    # Terminate the while success3 == False loop
                            success = True     # Terminate the while success == False loop
                            # print('Successfully found physical solutions!')
                    except ValueError:
                        success3 = False
                        # print('Oops! Have to try another initial guess...')
        return pop 

    def getExptvalue(self, pop: np.array, site: Site) -> float:
        """
        Calculate the expectation value of the number of particles at a given site at a given time
        Arguments:
            site {Site} -- site object
            pop {np.array} -- Population vector of the states
        """
        site_id = self.site2id[site]
        expt=0
        #loop through all the possible states
        for i in range(self.num_state):
            expt+=self.idx2state(i)[site_id]*pop[i]   #sum((number of particle)*(probability))

        return expt
    
    def population(self, pop: np.array, site: Site, occupancy: int) -> float:
        site_id = self.site2id[site]
        ans = 0
        for i in range(self.num_state):
            #Loop through all the possible states
            if self.idx2state(i)[site_id] == occupancy:   #For every state, the number of electrons on each site is known, (ex.)state[0]=[1 2 0 3 2...], state[1]=[0 2 3 1 ...]
                # It loops through all the states to find where the cof th element of (ex.)state:[0 1 1 0 2 3...] is equal to the given redox state
                # Population of electron at each cofactor = redox state of that cofactor
                ans += pop[i]

        return ans
    
    def getReservoirFlux(self, name: str, site: Site, pop: np.array) -> float:
        """
        Calculate the instantaneous net flux into the reservoir connected to the reservoir
        Arguments:
            reservoir_id {int} -- Reservoir ID
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux connected to the reservoir
        """
        site_id = self.site2id[site]
        reservoir_id = self.reservoir2id[name]
        name, energy, potential = self.reservoirInfo[reservoir_id]
        flux = 0
        for initial in range(self.num_state):
            for final in range(self.num_state):
                if self.idx2state(initial)[site_id] == 1 and self.idx2state(final)[site_id] == 0:
                    # initial, final state found! check other electron conservation
                    I = np.delete(self.idx2state(initial), [site_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                    J = np.delete(self.idx2state(final), [site_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                    if np.array_equal(I, J):
                        kf = self.K[final][initial]
                        kb = self.K[initial][final]
                        print(pop[initial], kf, pop[final], kb)
                        flux += pop[initial] * kf
                        flux -= pop[final] * kb

        return flux
