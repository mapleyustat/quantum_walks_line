###########################################################################################################################
# Title: Quantum Walks on an integer number line.
# Date: 02/20/2019, Wednesday
# Author: Minwoo Bae (minwoo.bae@uconn.edu)
# Institute: The Department of Computer Science and Engineering, UCONN
###########################################################################################################################
import numpy as np
from numpy import sqrt
from numpy import transpose as T
from numpy import tensordot as tensor
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import dot
from numpy import outer
from numpy import array as vec
from numpy import reshape
from numpy import kron #Kronecker product (kron): matrix tensor matrix
from numpy import eye as id
from numpy import append
from numpy import insert
import matplotlib.pyplot as plt

# Hadamard operator:
H = (1/sqrt(2))*vec([[1, 1],[1, -1]])
h = vec([[1, 1],[1, -1]])

# COIN operators:
COIN = vec([[1,0],[0,1]])

# Creates a standard basis matrix:
def get_pos_space(time):
    t = time
    if t==0:
        dim = 2
        pos_basis = id(dim, dim, 0, dtype=float)
    else:
        dim = 2*abs(t)+1
        pos_basis = id(dim, dim, 0, dtype=float)
    return pos_basis

# Creates nxn standard matrix:
def get_id_mat(time):
    t = time
    if t==0:
        dim = 2
        id_mat = id(dim, dim, 0, dtype=float)
    else:
        dim = 2*abs(t)+1
        id_mat = id(dim, dim, 0, dtype=float)
    return id_mat

# Creates an init state:
def get_init_state(time, type):

    t = time
    typ = type

    basis = get_pos_space(t)
    p0 = basis[t]

    if type==0:
        # COIN == head and state==0
        Hc = COIN[typ]
        return kron(Hc, p0)
    elif type==1:
        # COIN == tail and state==0
        Tc = COIN[typ]
        return kron(Tc, p0)
    else:
        # COIN == head + tail/sqrt(2) and state==0
        Hc = COIN[0]
        Tc = COIN[1]
        p0 = basis[0]
        quantum_coin = (Hc + Tc)/sqrt(2)
        return kron(quantum_coin, p0)

# Creates the Unitary operator:
def get_unitary_operator(time):

    t = time

    basis = get_pos_space(t)

    I = get_id_mat(t)
    H_tensor_I = kron(H, I)

    c_dim = len(COIN)
    p_dim = len(basis)

    head_temp = []
    tail_temp = []
    S_up = np.zeros([2*p_dim, 2*p_dim])
    S_down = np.zeros([2*p_dim, 2*p_dim])
    S = np.zeros([2*p_dim, 2*p_dim])

    for i in range(c_dim):

        for j in range(1, p_dim, 2):

            # COIN == head
            if i == 0:
                head_temp = kron(outer(COIN[i], COIN[i]), outer(basis[j+1], basis[j]))
                S_up += head_temp
                head_temp = []

            # COIN == tail
            else:
                tail_temp = kron(outer(COIN[i], COIN[i]), outer(basis[j-1], basis[j]))
                S_down += tail_temp
                tail_temp = []

    S = S_up + S_down

    U = S.dot(H_tensor_I)
    return U

def get_quantum_walk_state(time, init_type):

    t = time
    curr_state = []
    typ = init_type

    for i in range(t):
        if i == 0:
            p0 = get_init_state(i+1, typ)
            curr_state = get_unitary_operator(i+1).dot(p0)
            print('current state:', i+1)
            print(curr_state)
        else:
            print('current state:', i+1)
            n = len(curr_state)
            print(curr_state)

            temp_a = curr_state[:n/2]
            temp_b = curr_state[n/2:]

            na = len(temp_a)
            nb = len(temp_b)

            temp_a = insert(temp_a, na, 0, axis=0)
            temp_a = insert(temp_a, 0, 0, axis=0)
            temp_b = insert(temp_b, nb, 0, axis=0)
            temp_b = insert(temp_b, 0, 0, axis=0)

            curr_state = append(temp_a, temp_b)
            U = get_unitary_operator(i+1)
            curr_state = U.dot(curr_state)
            print(curr_state)

    return curr_state

trials = 70
type = 0
qWalkState = get_quantum_walk_state(trials, type)
nq = len(qWalkState)

if type == 1:
    qWalkTemp = qWalkState[nq/2:]
    print(qWalkTemp**2)

else:
    qWalkTemp = qWalkState[:nq/2]
    # print(qWalkState**2)
    print(qWalkTemp**2)

plt.plot(qWalkTemp**2)
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.show()