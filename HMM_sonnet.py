########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import math
import scipy.special

debug = False

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = np.array([1. / self.L for _ in range(self.L)])

    def prefix_probability(self, log_probs, prefix_len, last_state, 
                            new_state, new_output):
        '''
        Helper function for the viterbi algorithm.
        '''
        if debug:
            print(f"\tCalling prefix_probability with prefix_len = {prefix_len}, last_state = {last_state}, new_state = {new_state}, new_output = {new_output}")
        prev_prob       = log_probs[prefix_len, last_state]
        if debug:
            print(f"prev_prob({prefix_len}, {last_state}): {np.exp(prev_prob)}")
        transition_prob = self.A[last_state, new_state]
        if debug:
            print(f"transition_prob({last_state} to {new_state}): {transition_prob}")
        output_prob     = self.O[new_state, new_output]
        if debug:
            print(f"output_prob({new_state}, {new_output}): {output_prob}")
            print(f"returning: {prev_prob + np.log(transition_prob) + np.log(output_prob)}")
        return prev_prob + np.log(transition_prob) + np.log(output_prob)
               # (math.log(transition_prob) if transition_prob != 0.0 else 0) + \
               # (math.log(output_prob) if output_prob != 0.0 else 0)

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0
        log_probs = np.array([[0. for _ in range(self.L)] for _ in range(M + 1)])
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Base case is special! 
        log_probs[0,:] = [np.log(self.A_start[state] * self.O[state, x[0]]) 
                            for state in range(self.L)]
        seqs[0]  = [str(i) for i in range(self.L)]

        # "Recursive" case
        for i in range(1, M):
            log_probs[i,:] = [
                max([self.prefix_probability(log_probs, i-1, last_end, new_end, x[i]) 
                      for last_end in range(self.L)]) 
                for new_end in range(self.L)]
            most_likely_prev_states = [seqs[i-1][np.argmax([
                            self.prefix_probability(log_probs, i-1, last_end, 
                                                new_end, x[i]) 
                    for last_end in range(self.L)])] for new_end in range(self.L)]
            seqs[i]  = [most_likely_prev_states[new_end] + str(new_end)
                          for new_end in range(self.L)]
            if debug:
                raise Exception()


        max_seq = seqs[-2][np.argmax(log_probs[-2,:])]
        if debug:
            print(f"x: {x}")
            print("seqs:")
            print(seqs)
            print("probs:")
            print(np.exp(log_probs))
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        # M = len(x)      # Length of sequence.
        # alphas = np.zeros((M, self.L))

        # # Base case is special.
        # alphas[0,:] = self.A_start * self.O[:,x[0]]

        # # "Recursive" case
        # for i in range(1, M):
        #     sum_term = [(sum([alphas[i-1,j]*self.A[j,z] 
        #                     for j in range(self.L)])) 
        #                 for z in range(self.L)]
        #     sum_term = np.array(sum_term)
            
        #     alphas[i,:] = self.O[:,x[i]] * sum_term
        #     if normalize:
        #         alphas[i,:] /= alphas[i,:].sum()
        #         assert(np.abs(alphas[i,:].sum() - 1.0)<1e-9)
        # if debug:
        #     print(alphas)

        # return alphas

        M = len(x)      # Length of sequence.
        alphas = np.zeros((M, self.L))

        # Base case is special.
        alphas[0,:] = self.A_start * self.O[:,x[0]]

        # "Recursive" case
        for i in range(1, M):
            sum_term = [(sum([alphas[i-1,j]*self.A[j,z] 
                            for j in range(self.L)])) 
                        for z in range(self.L)]
            sum_term = np.array(sum_term)
            alphas[i,:] = self.O[:,x[i]] * sum_term
            if normalize:
                alphas[i,:] /= alphas[i,:].sum()
        if debug:
            print(alphas)

        return alphas

        # M = len(x)      # Length of sequence.
        # log_alphas = np.zeros((M, self.L))

        # # Base case is special.
        # log_alphas[0,:] = np.log(self.A_start) + np.log(self.O[:,x[0]])

        # # "Recursive" case
        # for i in range(1, M):
        #     sum_term = [(sum([np.exp(log_alphas[i-1,j])*self.A[j,z] 
        #                     for j in range(self.L)])) 
        #                 for z in range(self.L)]
        #     sum_term = np.log(np.array(sum_term))
        #     log_alphas[i,:] = np.log(self.O[:,x[i]]) + sum_term
        # if debug:
        #     print(log_alphas)

        # return np.exp(log_alphas)


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = np.ones((M, self.L))

        # Base case is initialized at one already.

        # Recursive case
        for i in range(M-2, -1, -1):
            for z in range(self.L):
                betas[i, z] = sum([betas[i+1,j] * self.A[z,j] * self.O[j, x[i+1]] 
                    for j in range(self.L)])
            if normalize:
                betas[i,:] /= betas[i,:].sum()

        if debug:
            print(betas)

        return betas

        # M = len(x)      # Length of sequence.
        # log_betas = np.zeros((M, self.L))

        # # Base case is initialized at zero already.

        # # Recursive case
        # for i in range(M-2, -1, -1):
        #     for z in range(self.L):
        #         log_betas[i, z] = sum([np.exp(log_betas[i+1,j]) * self.A[z,j] * self.O[j, x[i+1]] 
        #             for j in range(self.L)])
        #     log_betas[i,:] = np.log(log_betas[i,:])

        # if debug:
        #     print(log_betas)

        # return np.exp(log_betas)


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''
        self.A = np.zeros((self.L, self.L))
        self.O = np.zeros((self.L, self.D))

        # Calculate each element of A using the M-step formulas.
        if debug:
            print(X)
            print(Y)
        state_transition_num = np.zeros(self.L)
        for seq in Y:
            last_state = None
            for state in seq:
                if last_state != None:
                    self.A[last_state, state] += 1
                    state_transition_num[last_state] += 1
                last_state = state
        for i in range(self.L):
            self.A[i,:] /= state_transition_num[i]
        if debug:
            print("state_transition_num:")
            print(state_transition_num)


        # Calculate each element of O using the M-step formulas.
        state_occurrences = np.zeros(self.L)
        for seq_num, seq in enumerate(X):
            for i, token in enumerate(seq):
                state = Y[seq_num][i]
                self.O[state, token] += 1
                state_occurrences[state] += 1
        for i in range(self.L):
            self.O[i,:] /= state_occurrences[i]
        if debug:
            print("state_occurrences")
            print(state_occurrences)


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        # A and O are already initialized, need to numpyfy them..
        self.A = np.array(self.A)
        self.O = np.array(self.O)

        for i in range(N_iters):
            print(f"\rTraining iteration {i+1}/{N_iters}....    ", end = "")
            # each alpha/beta has shape (len(seq) x L)
            all_alphas = []
            all_betas  = []
            for seq in X:
                alphas = self.forward(seq, normalize = True)
                betas  = self.backward(seq, normalize = True)
                all_alphas.append(alphas)
                all_betas.append(betas)

            # Calculate marginal probabilities of y. 
            # Shape is n_seqs x len(seq) x L
            # Also calculate marginal probabilities, which has shape 
            #          n_seqs x len(seq) x L(a) x L(b)
            y_probs = []
            marginal_probs = []
            for seq_num, seq in enumerate(X):
                y_probs.append(-1*np.ones((len(seq), self.L)))
                marginal_probs.append(-1*np.ones((len(seq), self.L, self.L)))
                for i in range(len(seq)):
                    norm_constant = 0
                    marg_norm_constant = 0
                    for z in range(self.L):
                        term = all_alphas[seq_num][i,z] * all_betas[seq_num][i,z]
                        norm_constant += term
                        y_probs[-1][i,z] = term
                        if i == 0:
                            continue
                        for b in range(self.L):
                            marg_term = all_alphas[seq_num][i-1,z]*self.A[z,b]\
                                *self.O[b,X[seq_num][i]]*all_betas[seq_num][i,b]
                            marginal_probs[seq_num][i,z,b] = marg_term
                            marg_norm_constant += marg_term
                    y_probs[-1][i,:] /= norm_constant
                    marginal_probs[-1][i,:,:] /= marg_norm_constant

            # Re-estimate A
            for a in range(self.L):
                for b in range(self.L):
                    self.A[a,b] = sum([np.sum(np.array([marginal_probs[seq_idx][i,a,b] 
                                                        for i in range(1, len(seq))]))
                                       for seq_idx, seq in enumerate(X)])

            # Normalize A
            for a in range(self.L):
                self.A[a,:] /= sum([y_probs[seq_num][:-1,a].sum() for seq_num in range(len(X))])

            # Re-estimate O
            self.O = np.zeros(self.O.shape)
            for seq_idx, seq in enumerate(X):
                for i, token in enumerate(seq):
                    for z in range(self.L):
                        self.O[z, token] += y_probs[seq_idx][i,z]
            # Normalize O
            for z in range(self.L):
                self.O[z, :] /= sum([np.sum(np.array([y_probs[seq_idx][i,z] 
                                           for i in range(len(seq))]))
                                          for seq_idx, seq in enumerate(X)])
                

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = -1 * np.ones(M, dtype = np.int)
        states = -1 * np.ones(M, dtype = np.int)

        states[0]   = np.random.choice(self.L)
        emission[0] = int(np.random.choice(self.D, p = self.O[states[0],:]))

        for i in range(1, M):
            states[i]   = int(np.random.choice(self.L, p = self.A[states[i-1],:]))
            emission[i] = int(np.random.choice(self.D, p = self.O[states[i],:]))

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x, normalize = True)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[0][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters, manual_states=False):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    epsilon = 1e-4
    if manual_states:
        assert n_states == 4
        A = [[0. for i in range(L)] for j in range(L)]
        A[0][0] = epsilon
        A[0][1] = epsilon
        A[1][2] = epsilon
        A[1][3] = epsilon
        A[2][0] = epsilon
        A[2][1] = epsilon
        A[3][2] = epsilon
        A[3][3] = epsilon
        A[0][2] = np.random.random()*(1-(2*epsilon))
        A[0][3] = 1 - A[0][2] - 2*epsilon
        A[2][2] = np.random.random()*(1-(2*epsilon))
        A[2][3] = 1 - A[2][2] - 2*epsilon
        A[1][0] = np.random.random()*(1-(2*epsilon))
        A[1][1] = 1 - A[1][0] - 2*epsilon
        A[3][0] = np.random.random()*(1-(2*epsilon))
        A[3][1] = 1 - A[3][0] - 2*epsilon
    else:
        A = [[random.random() for i in range(L)] for j in range(L)]

        for i in range(len(A)):
            norm = sum(A[i])
            for j in range(len(A[i])):
                A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
