import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
import numpy as np
import copy
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("faulty alarm","alarm")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("gauge", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    t_cpd = TabularCPD("temperature", 2, values=[[0.80],[0.20]])
    fa_cpd = TabularCPD("faulty alarm", 2, values=[[0.85],[0.15]])
    fg_cpd = TabularCPD("faulty gauge", 2, values=[[0.95,0.2], [0.05, 0.8]], evidence=['temperature'], evidence_card=[2])
    g_cpd = TabularCPD("gauge", 2, values=[[0.95, 0.2, 0.05, 0.8], \
                       [0.05, 0.8, 0.95, 0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])
    a_cpd = TabularCPD("alarm", 2, values=[[0.9,0.55,0.1,0.45], \
                       [0.1, 0.45, 0.90, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2,2])
    bayes_net.add_cpds(t_cpd, fa_cpd, fg_cpd, g_cpd, a_cpd)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal
    probability of the alarm
    ringing in the
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'])
    alarm_prob = marginal_prob['alarm'].values[1]
    # print(alarm_prob)
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge
    showing hot in the
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['gauge'])
    gauge_prob = conditional_prob['gauge'].values[1]
    # print(gauge_prob)
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty alarm':0, 'faulty gauge':0})
    temp_prob = conditional_prob['temperature'].values[1]
    # print(temp_prob)
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")

    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "BvC")
    BayesNet.add_edge("C", "CvA")

    skill_dist = [[0.15], [0.45],[0.30], [0.10]]
    a_cpd = TabularCPD("A", 4, values=skill_dist)
    b_cpd = TabularCPD("B", 4, values=skill_dist)
    c_cpd = TabularCPD("C", 4, values=skill_dist)

    game_dist = [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
                 [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                 [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]]

    # avb_cpd = TabularCPD("AvB", 3, values=game_dist, evidence=["A", "B"], evidence_card=[4, 4])
    avb_cpd = TabularCPD("AvB", 3, values=game_dist, evidence=["A", "B"], evidence_card=[4, 4])
    bvc_cpd = TabularCPD("BvC", 3, values=game_dist, evidence=["B", "C"], evidence_card=[4, 4])
    cva_cpd = TabularCPD("CvA", 3, values=game_dist, evidence=["C", "A"], evidence_card=[4, 4])

    BayesNet.add_cpds(a_cpd, b_cpd, c_cpd, avb_cpd, bvc_cpd, cva_cpd)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=["BvC"],evidence={"AvB": 0, "CvA": 2})
    posterior = conditional_prob["BvC"].values
    return posterior # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.
    """
    # TODO: finish this function
    # import pdb
    # pdb.set_trace()
    if initial_state is None or len(initial_state) is 0:
        initial_state = [-1, -1, -1, -1, -1, -1]
        initial_state[0], initial_state[1], initial_state[2] = random.choice([0,1,2,3]), random.choice([0,1,2,3]), random.choice([0,1,2,3])
        initial_state[3], initial_state[4], initial_state[5] = 0, random.choice([0,1,2]), 2
    else:
        initial_state = list(initial_state)

    game_dist = [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
                 [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                 [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]]
    skill_dist = [0.15, 0.45, 0.30, 0.10]

    update_var = random.choice([0,1,2,4])
    # update_var = 1

    if update_var <= 2:
        match_map = {0: [3, 5], 1: [4, 3], 2: [5, 4]}
        team_map  = {0: [1, 2], 1: [2, 0], 2: [0, 1]}
        skill_prob = [0,0,0,0]
        sum = 0.0
        match1_idx, match2_idx = match_map[update_var]
        opp1, opp2 = team_map[update_var]
        for s in range(4):
            match1_res, match2_res = initial_state[match1_idx], initial_state[match2_idx]
            opp1_res, opp2_res = initial_state[opp1], initial_state[opp2]
            match1_prob = game_dist[match1_res][s*4 +opp1_res]
            match2_prob = game_dist[match2_res][opp2_res*4 + s]
            skill_prob[s] = skill_dist[s] * match1_prob * match2_prob
            sum += skill_prob[s]
        # Normalize
        for s in range(4): skill_prob[s] /= sum
        # Sample
        new_val = np.random.choice(4, 1, p=skill_prob)[0]
    else:
        skill_map = {3: [0, 1], 4: [1, 2], 5: [2, 0]}
        t1, t2 = skill_map[update_var]
        t1_val, t2_val = initial_state[t1], initial_state[t2]
        res_prob = [0, 0, 0]
        sum = 0.0
        for r in range(3):
            res_prob[r] = game_dist[r][t1_val*4 + t2_val]
            sum += res_prob[r]
        # Normalize
        # for r in range(3): res_prob[r] /= sum
        # Sample
        new_val = np.random.choice(3, 1, p=res_prob)[0]

    initial_state[update_var] = new_val
    sample = tuple(initial_state)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    if initial_state is None or len(initial_state) is 0:
        initial_state = [-1, -1, -1, -1, -1, -1]
        initial_state[0], initial_state[1], initial_state[2] = random.choice([0,1,2,3]), random.choice([0,1,2,3]), random.choice([0,1,2,3])
        initial_state[3], initial_state[4], initial_state[5] = 0, random.choice([0,1,2]), 2
    else:
        initial_state = list(initial_state)

    game_dist = [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
                 [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                 [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]]
    skill_dist = [0.15, 0.45, 0.30, 0.10]

    theta_star = [-1, -1, -1, -1, -1, -1]

    theta_star[0], theta_star[1], theta_star[2] = random.choice([0,1,2,3]), random.choice([0,1,2,3]), random.choice([0,1,2,3])
    theta_star[3], theta_star[4], theta_star[5] = 0, random.choice([0,1,2]), 2
    # import pdb
    # pdb.set_trace()

    alpha = min(1, _find_joint_prob(theta_star, game_dist, skill_dist)/_find_joint_prob(initial_state, game_dist, skill_dist))
    if _random_choice(alpha):
        sample = tuple(theta_star)
    else:
        sample = tuple(initial_state)

    return sample

def _random_choice(p):
    if random.random() < p:
        return 1
    return 0

def _find_joint_prob(state, game_dist, skill_dist):
    prob = 1.0
    for i in range(3):
        prob *= skill_dist[state[i]]
    prob *= game_dist[state[3]][state[0]*4 + state[1]]
    prob *= game_dist[state[4]][state[1]*4 + state[2]]
    prob *= game_dist[state[5]][state[2]*4 + state[0]]
    return prob

def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    # import pdb
    # pdb.set_trace()
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    Gibbs_curr_prob, Gibbs_prev_prob = [0,0,0], [0,0,0]
    MH_curr_prob, MH_prev_prob = [0,0,0], [0,0,0]
    # TODO: finish this function
    # raise NotImplementedError
    # posterior = calculate_posterior(bayes_net)
    delta, N = 0.0001, 1000
    burn_in = 100000
    gibbs_converged_count, gibbs_consec = 0, False
    if initial_state is None: state = []
    else: state = list(initial_state)
    for i in range(burn_in): state = Gibbs_sampler(bayes_net, state)
    for i in range(100000):
        # print(Gibbs_convergence, Gibbs_count)
        state = Gibbs_sampler(bayes_net, state)
        Gibbs_count += 1
        Gibbs_convergence[state[4]] += 1
        sum_gibbs = 0.0
        for i in range(3): sum_gibbs += Gibbs_convergence[i]
        for i in range(3): Gibbs_curr_prob[i] = Gibbs_convergence[i] / sum_gibbs
        if _find_diff(Gibbs_curr_prob, Gibbs_prev_prob) <= delta:
            gibbs_converged_count += 1
            gibbs_consec = True
        else:
            gibbs_consec = False
            gibbs_converged_count = 0
        if gibbs_converged_count >= N and gibbs_consec: break
        Gibbs_prev_prob = copy.copy(Gibbs_curr_prob)
    sum = 0.0
    for i in range(len(Gibbs_convergence)): sum += Gibbs_convergence[i]
    for i in range(len(Gibbs_convergence)): Gibbs_convergence[i] /= sum
    # print(Gibbs_convergence, Gibbs_count)
    # print(posterior)

    mh_converged_count, mh_consec = 0, False
    if initial_state is None: state = []
    else: state = list(initial_state)
    for i in range(burn_in): state = MH_sampler(bayes_net, state)
    for i in range(1000000):
        state_new = MH_sampler(bayes_net, state)
        if state_new == state: MH_rejection_count += 1
        MH_convergence[state_new[4]] += 1

        sum_mh = 0.0
        for i in range(3): sum_mh += MH_convergence[i]
        for i in range(3): MH_curr_prob[i] = MH_convergence[i] / sum_mh
        if _find_diff(MH_curr_prob, MH_prev_prob) <= delta:
            mh_converged_count += 1
            mh_consec = True
        else:
            mh_consec = False
            mh_converged_count = 0

        if mh_converged_count >= N and mh_consec: break
        MH_prev_prob = copy.copy(MH_curr_prob)

        state = state_new
        MH_count += 1
    sum = 0.0
    for i in range(len(MH_convergence)): sum += MH_convergence[i]
    for i in range(len(MH_convergence)): MH_convergence[i] /= sum
    # print(MH_convergence, MH_rejection_count/MH_count, MH_count)
    # print(posterior)

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def _find_diff(p1, p2):
    sum1 = 0.0
    for i in range(3): sum1 += abs(p1[i]-p2[i])
    # print(p1, p2, sum)
    return sum1

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.25
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Advait Koparkar"
