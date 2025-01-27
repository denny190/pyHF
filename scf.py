import numpy as np
from utils import h_core, s_orthogonalized, init_density_matrix_hcore, init_density_matrix_zeros, construct_fock_matrix

def diagonalize_fock_matrix(F, S_half_inv):
    F_prime = S_half_inv @ F @ S_half_inv 
    eigvals, eigvecs = np.linalg.eigh(F_prime)
    C = S_half_inv @ eigvecs
    return C, eigvals

def construct_density_matrix(C, nb, z, charge):
    P_new = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            electrons = sum(z) - charge
            occ_orbitals = electrons // 2  # For closed-shell systems
            for a in range(occ_orbitals):
                P_new[i, j] += 2 * C[i, a] * C[j, a]
    return P_new
    
def electronic_energy(P, H, F, nb):
    E_elec = 0
    for i in range(nb):
        for j in range(nb):
                E_elec += P[i, j] * (H[i,j] + F[i,j])
    return 0.5 * E_elec

def scf(T, V, S, TEI, nb, z, charge, max_iterations=100, convergence_threshold=1e-6):
    H = h_core(T, V)
    So = s_orthogonalized(S)
    #P = init_density_matrix_zeros(nb)
    P = init_density_matrix_hcore(H, S, nb, charge)
    E_old = 0  
    energies = []

    for iteration in range(max_iterations):
        F = construct_fock_matrix(H, P, TEI, nb)
        C, eigvals = diagonalize_fock_matrix(F, So)
        P_new = construct_density_matrix(C, nb, z, charge)
        E_elec = electronic_energy(P_new, H, F, nb)
        energies.append(E_elec)

        print(f"Iteration {iteration + 1}: E_elec = {E_elec}")

        if abs(E_elec - E_old) < convergence_threshold:
            print(f"SCF converged after {iteration + 1} iterations")
            break

        P = P_new
        E_old = E_elec
    else:
        print("SCF did not converge")

    return E_elec, P, C, energies
