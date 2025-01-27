import numpy as np
from scipy.linalg import fractional_matrix_power
import math

def read_input(input_file):
    input=open(input_file,"r")
    angtobohr = 0.5291772109
    natoms=0
    z=[]
    i=0
    labels=[]
    charge=0

    for index,line in enumerate(input): 
        if index==1:
            natoms=int(line.split()[0]) 
            coords=np.zeros((natoms,3))
        if index==3:
            charge=int(line.split()[0])
        if index > 3 and index < (natoms+4):
            aux=line.split()
            atomic_name=aux[0]    
            if atomic_name == "H":
                atomic_num = 1
            if atomic_name == "He":
                atomic_num = 2
            coord=[float(aux[1])/angtobohr,float(aux[2])/angtobohr,float(aux[3])/angtobohr]
            labels.append(atomic_name) #we add the data (atom names) to a list
            z.append(atomic_num) # we add the data (atomic number) to a list
            coords[i,:]=coord  #we add the coordinates of each atom
            i=i+1
    return natoms,charge,labels,z,coords

def h_core(T, V):
    return T + V


def s_orthogonalized(S):
    return fractional_matrix_power(S, -0.5)


def init_density_matrix_zeros(nb):
    return np.zeros((nb, nb))

def init_density_matrix_hcore(H, S, nb, charge):
    from scipy.linalg import fractional_matrix_power
    S_ortho = fractional_matrix_power(S, -0.5)
    H_prime = S_ortho @ H @ S_ortho

    eigvals, eigvecs = np.linalg.eigh(H_prime)

    C_core = S_ortho @ eigvecs

    num_electrons = nb - charge
    num_occupied = num_electrons // 2

    P = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            for a in range(num_occupied):
                P[i, j] += 2 * C_core[i, a] * C_core[j, a]

    return P


def construct_fock_matrix(H, P, TEI, nb):
    F = np.copy(H)
    for i in range(nb):
        for j in range(nb):
            for k in range(nb):
                for l in range(nb):
                    F[i, j] += P[k, l] * TEI[i, j, k, l]  # Coulomb term
                    F[i, j] -= 0.5 * P[k, l] * TEI[i, k, j, l]  # Exchange term
    return F


def boys_function(r):
    if r < 1e-6:
        return 1.0
    return 0.5 * math.sqrt(math.pi / r) * math.erf(math.sqrt(r))


def nuclear_repulsion_energy(atomic_numbers, coords):
    e_repulsion = 0.0
    for i in range(len(atomic_numbers)):
        for j in range(i + 1, len(atomic_numbers)):
            r_ij = np.linalg.norm(coords[i] - coords[j])
            e_repulsion += atomic_numbers[i] * atomic_numbers[j] / r_ij
    return e_repulsion
