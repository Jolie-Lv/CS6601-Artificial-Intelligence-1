
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: notebook.ipynb

import numpy as np
import operator

def part_1_a():
    """Provide probabilities for the word HMMs outlined below.
    Word BUY, CAR, and HOUSE.
    Review Udacity Lesson 8 - Video #29. HMM Training
    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)
        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """

    # TODO: complete this function.
    # raise NotImplementedError()

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0., 'Bend': 0.},
        'B2': {'B1': 0., 'B2': 0.625, 'B3': 0.375, 'Bend': 0.},
        'B3': {'B1': 0., 'B2': 0.0, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 1.0},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.75, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.0,
        'C3': 0.0,
        'Cend': 0.0,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0., 'Cend': 0.},
        'C2': {'C1': 0., 'C2': 0.0, 'C3': 1.0, 'Cend': 0.},
        'C3': {'C1': 0., 'C2': 0., 'C3': 0.8, 'Cend': 0.2},
        'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 1.0},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.7),
        'C3': (44.2, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.0,
        'H3': 0.0,
        'Hend': 0.0,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0., 'Hend': 0.},
        'H2': {'H1': 0., 'H2': 0.857, 'H3': 0.143, 'Hend': 0.},
        'H3': {'H1': 0., 'H2': 0., 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 1.0},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2':  (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x.
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if list(para_tuple) == [None, None]:
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.
    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).
        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend', 'H1', 'H2', 'H3', 'Hend']
        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}
        transition_probs (dict): dictionary representing transitions from each
                                 state to every other valid state such as for the above
                                 states, there won't be a transition from 'B1' to 'H1'
        emission_paras (tuple): parameters of Gaussian distribution
                                from each state.
    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )
    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.
    """

    sequence = []
    probability = 0.0
    K = len(states)
    T = len(evidence_vector)

    if K == 0 or T == 0:
        return ([], 0)

    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T), dtype=np.int32)
    sequence = [None] * T

    # init
    for i, state in enumerate(states):
        T1[i, 0] = prior_probs[state] * gaussian_prob(evidence_vector[0], emission_paras[state])

    for i, evidence in enumerate(evidence_vector):
        for j, state in enumerate(states):
            if i == 0: continue

            t1, t2 = float("-inf"), None
            for k, s in enumerate(states):
                if state in transition_probs[s]:
                    v = T1[k, i-1] * transition_probs[s][state] * gaussian_prob(evidence_vector[i], emission_paras[state])
                    if v > t1:
                        t1, t2 = v, k
            T1[j, i], T2[j, i] = t1, t2

    probability, zT = float("-inf"), None

    if probability == 0.0:
        return ([], 0.0)

    for k, t in enumerate(T1[:, T-1]):
        if t > probability:
            probability, zT = t, k

    sequence[T-1] = states[zT]
    for i in range(T-1, 0, -1):
        z_p = T2[zT, i]
        sequence[i-1] = states[z_p]
        zT = z_p

    return sequence, probability

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.
    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimension transition &
    emission probabilities.
    """


    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.7), 'B2': (0.375, 0.3), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0.625, 0.05), 'B3': (0.375, 0.95), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125,0.091), 'H1': (0.125,0.091)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (1.0, 1.0)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.75, 2.773), (108.2, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.7), 'C2': (0.333, 0.3), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0.0, 0.625), 'C3': (1.0, 0.375), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.8, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125), 'H1': (0.067, 0.125)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (1.0, 1.0)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.30, 10.659)],
        'C2': [(43.667, 1.7), (37.110, 4.306)],
        'C3': [(44.2, 7.341), (50.00, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.7), 'H2': (0.333, 0.3), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'C1': (0.063, 0.059), 'B1': (0.063, 0.059)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (1.0, 1.0)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.
    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    Evidence vector is a list of tuples where the first element of each tuple is the right
    hand coordinate and the second element is the left hand coordinate.
    """

    sequence = []
    probability = 0.0
    T = len(evidence_vector)
    K = len(states)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T), dtype=np.int32)
    sequence = [None] * T

    if K == 0 or T == 0:
        return ([], 0)

    # init
    for i, state in enumerate(states):
        pR = prior_probs[state] * gaussian_prob(evidence_vector[0][0], emission_paras[state][0])
        pL = gaussian_prob(evidence_vector[0][1], emission_paras[state][1])
        T1[i, 0] = pR * pL


    for i, evidence in enumerate(evidence_vector):
        for j, state in enumerate(states):
            if i == 0: continue

            t1, t2 = float("-inf"), None
            for k, s in enumerate(states):
                if state in transition_probs[s]:
                    pR = T1[k, i-1] * transition_probs[s][state][0] * gaussian_prob(evidence_vector[i][0], emission_paras[state][0])
                    pL = transition_probs[s][state][1] * gaussian_prob(evidence_vector[i][1], emission_paras[state][1])
                    v = pL * pR
                    if v > t1:
                        t1, t2 = v, k
            T1[j, i], T2[j, i] = t1, t2

    probability, zT = float("-inf"), None

    if probability == 0.0:
        return ([], 0.0)

    for k, t in enumerate(T1[:, T-1]):
        if t > probability:
            probability, zT = t, k

    if probability == 0.0:
        return ([], 0.0)

    sequence[T-1] = states[zT]
    for i in range(T-1, 0, -1):
        z_p = T2[zT, i]
        sequence[i-1] = states[z_p]
        zT = z_p

    return sequence, probability

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def return_your_name():
    """Return your name
    """
    return "Advait Koparkar"

def MLLR_results():

    """Complete the MLLR adaptation process with the new adaptation data and
     return the new emission parameters for each state.
    """
    # TODO: complete this function.

#     b_emission_paras = {
#         'B1': [(41.75, 2.773), (108.2, 17.314)],
#         'B2': [(58.625, 5.678), (78.670, 1.886)],
#         'B3': [(53.125, 5.418), (64.182, 5.573)],
#         'Bend': [(None, None), (None, None)]
#     }

# (61,123), (61, 116), (59, 121), (65, 99), (73, 97), (75, 98), (79, 74), (79, 84), (79, 84), (74, 89), (68, 81)

    # X_b, Y_b
    X_b = np.array([[1,     1,     1,     1,    1,     1,     1,   1,   1,   1,   1],
                    [41.75, 41.75, 41.75,58.625,58.625,58.625,53.125,53.125,53.125,53.125,53.125],
                    [108.2, 108.2, 108.2,78.670,78.670,78.670,64.182,64.182,64.182,64.182,64.182]])

    Y_b = np.array([[61,61,59,65,73,75,79,79,79,74,68],
                    [123,116,121,99,97,98,74,84,84,89,81]])

    mu_b = np.array([[1,1,1],
                     [41.75,58.625,53.125],
                     [108.2,78.670,64.182]])
#     import pdb
#     pdb.set_trace()
    mu_b_new = Y_b.dot((X_b.T)).dot(np.linalg.inv(X_b.dot(X_b.T))).dot(mu_b)

#     c_emission_paras = {
#         'C1': [(35.667, 4.899), (56.30, 10.659)],
#         'C2': [(43.667, 1.7), (37.110, 4.306)],
#         'C3': [(44.2, 7.341), (50.00, 7.826)],
#         'Cend': [(None, None), (None, None)]
#     }
# (44, 73), (53, 70), (62, 78), (64, 62), (66, 58), (59, 51), (61, 76), (58, 90)
    X_c = np.array([[1,     1,     1,     1,    1,     1,     1,   1],
                    [43.667,43.667,43.667,43.667,43.667,43.667,44.2,44.2],
                    [37.110,37.110,37.110,37.110,37.110,37.110,50.00,50.00]])

    Y_c = np.array([[44,53,62,64,66,59,61,58],
                    [73,70,78,62,58,51,76,90]])

    mu_c = np.array([[1,1,1],
                     [35.667,43.667,44.2],
                     [56.30,37.110,50.00]])
    mu_c_new = Y_c.dot(X_c.T).dot(np.linalg.inv(X_c.dot(X_c.T))).dot(mu_c)
#     h_emission_paras = {
#         'H1': [(45.333, 3.972), (53.600, 7.392)],
#         'H2': [(34.952, 8.127), (37.168, 8.875)],
#         'H3': [(67.438, 5.733), (74.176, 8.347)],
#         'Hend': [(None, None), (None, None)]
#     }
# (59, 65), (59, 68), (60, 69), (57, 70), (56, 64), (49, 59), (51, 57), (51, 51), (53, 51), (59, 59), (72, 79), (81, 82), (82, 89), (84, 90), (86, 90), (90, 93)

    X_h = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [45.333,45.333,45.333,45.333,45.333,34.952,34.952,34.952,34.952,34.952,67.438,67.438,67.438,67.438,67.438,67.438],
                    [53.600,53.600,53.600,53.600,53.600,37.168,37.168,37.168,37.168,37.168,74.176,74.176,74.176,74.176,74.176,74.176]])

    Y_h = np.array([[59,59,60,57,56,49,51,51,53,59,72,81,82,84,86,90],
                    [65,68,69,70,64,59,57,51,51,59,79,82,89,90,90,93]])

    mu_h = np.array([[1,1,1],
                     [45.333,34.952,67.438],
                     [53.600,37.168,74.176]])

    mu_h_new = Y_h.dot(X_h.T).dot(np.linalg.inv(X_h.dot(X_h.T))).dot(mu_h)

#     print(X_b.shape, Y_b.shape, X_c.shape, Y_c.shape, X_h.shape, Y_h.shape)
#     print(mu_b_new.shape, mu_c_new.shape, mu_h_new.shape)

    b_emission_paras = {
        'B1': [(mu_b_new[0,0], 2.773), (mu_b_new[1,0], 17.314)],
        'B2': [(mu_b_new[0,1], 5.678), (mu_b_new[1,1], 1.886)],
        'B3': [(mu_b_new[0,2], 5.418), (mu_b_new[1,2], 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    c_emission_paras = {
        'C1': [(mu_c_new[0,0], 4.899), (mu_c_new[1,0], 10.659)],
        'C2': [(mu_c_new[0,1], 1.7), (mu_c_new[1,1], 4.306)],
        'C3': [(mu_c_new[0,2], 7.341), (mu_c_new[1,2], 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    h_emission_paras = {
        'H1': [(mu_h_new[0,0], 3.972), (mu_h_new[1,0], 7.392)],
        'H2': [(mu_h_new[0,1], 8.127), (mu_h_new[1,1], 8.875)],
        'H3': [(mu_h_new[0,2], 5.733), (mu_h_new[1,2], 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_emission_paras,
            c_emission_paras,
            h_emission_paras)