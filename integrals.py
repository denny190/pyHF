import numpy as np
import os
import json
from utils import boys_function

def build_basis(natoms, atomic_numbers, coordinates, basis_set_name, basis_directory="basis/"):
    bfno = 0
    basis_set = []

    basis_file = os.path.join(basis_directory, f"{basis_set_name}.json")
    if not os.path.exists(basis_file):
        raise FileNotFoundError(f"Basis set file not found: {basis_file}")

    with open(basis_file, "r") as file:
        basis_data = json.load(file)

    # basis set JSON's contain all info about atom constants that we need, so no need to maintain a separate periodic table
    for atom_idx, atomic_number in enumerate(atomic_numbers):
        if str(atomic_number) not in basis_data["elements"]:
            raise ValueError(f"Basis set not defined for atomic number {atomic_number}")

        element_basis = basis_data["elements"][str(atomic_number)]

        for shell in element_basis["electron_shells"]:
            angular_momentum = shell["angular_momentum"]
            exponents = [float(e) for e in shell["exponents"]]
            coefficients = [[float(c) for c in coeff_set] for coeff_set in shell["coefficients"]]

            for ang_mom_idx, ang_mom in enumerate(angular_momentum):
                basis_set.append(
                    [
                        bfno,  
                        atom_idx,
                        len(exponents), 
                        exponents,  
                        coefficients[ang_mom_idx],
                        ang_mom, 
                    ]
                )
                bfno += 1

    return bfno, basis_set

def compute_one_electron_integrals(num_atoms, num_basis, atomic_numbers, coordinates, basis_set):
    overlap_matrix = np.zeros((num_basis, num_basis))
    kinetic_energy_matrix = np.zeros((num_basis, num_basis))
    nuclear_attraction_matrix = np.zeros((num_basis, num_basis))

    for basis_fn1 in range(num_basis):
        atom1_idx = basis_set[basis_fn1][1]
        atom1_coords = coordinates[atom1_idx, :]

        for basis_fn2 in range(num_basis):
            atom2_idx = basis_set[basis_fn2][1]
            atom2_coords = coordinates[atom2_idx, :]

            for prim1_idx in range(basis_set[basis_fn1][2]):
                zeta1 = basis_set[basis_fn1][3][prim1_idx]
                coeff1 = basis_set[basis_fn1][4][prim1_idx]

                for prim2_idx in range(basis_set[basis_fn2][2]):
                    zeta2 = basis_set[basis_fn2][3][prim2_idx]
                    coeff2 = basis_set[basis_fn2][4][prim2_idx]

                    p = zeta1 + zeta2
                    q = zeta1 * zeta2 / p
                    gaussian_center = (zeta1 * atom1_coords + zeta2 * atom2_coords) / p

                    distance_squared = np.linalg.norm(atom2_coords - atom1_coords) ** 2
                    gaussian_overlap = (
                        coeff1
                        * coeff2
                        * (np.pi / p) ** (3 / 2)
                        * np.exp(-q * distance_squared)
                    )

                    overlap_matrix[basis_fn1, basis_fn2] += gaussian_overlap

                    kinetic_contribution = q * (3.0 - 2.0 * q * distance_squared)
                    kinetic_energy_matrix[basis_fn1, basis_fn2] += kinetic_contribution * gaussian_overlap

                    attraction_term = 0.0
                    for nucleus_idx in range(num_atoms):
                        nucleus_coords = coordinates[nucleus_idx, :]
                        nucleus_distance_squared = np.linalg.norm(
                            nucleus_coords - gaussian_center
                        ) ** 2
                        boys_factor = boys_function(p * nucleus_distance_squared)
                        attraction_term += boys_factor * atomic_numbers[nucleus_idx]

                    nuclear_attraction_matrix[basis_fn1, basis_fn2] -= (
                        2.0 * attraction_term * np.sqrt(p / np.pi) * gaussian_overlap
                    )

    return overlap_matrix, kinetic_energy_matrix, nuclear_attraction_matrix


def compute_two_electron_integrals(num_basis, coordinates, basis_set):
    two_electron_integrals = np.zeros((num_basis, num_basis, num_basis, num_basis))
    normalization_factor = np.sqrt(2.0) * (np.pi ** 1.25)

    for basis_fn1 in range(num_basis):
        atom1_idx = basis_set[basis_fn1][1]
        atom1_coords = coordinates[atom1_idx, :]

        for basis_fn2 in range(basis_fn1 + 1):
            atom2_idx = basis_set[basis_fn2][1]
            atom2_coords = coordinates[atom2_idx, :]
            distance12_squared = np.linalg.norm(atom2_coords - atom1_coords) ** 2

            for basis_fn3 in range(num_basis):
                atom3_idx = basis_set[basis_fn3][1]
                atom3_coords = coordinates[atom3_idx, :]

                for basis_fn4 in range(basis_fn3 + 1):
                    atom4_idx = basis_set[basis_fn4][1]
                    atom4_coords = coordinates[atom4_idx, :]
                    distance34_squared = np.linalg.norm(atom4_coords - atom3_coords) ** 2

                    if basis_fn1 * (basis_fn1 + 1) // 2 + basis_fn2 < basis_fn3 * (basis_fn3 + 1) // 2 + basis_fn4:
                        continue

                    integral_value = 0.0
                    for prim1_idx in range(basis_set[basis_fn1][2]):
                        zeta1 = basis_set[basis_fn1][3][prim1_idx]
                        coeff1 = basis_set[basis_fn1][4][prim1_idx]

                        for prim2_idx in range(basis_set[basis_fn2][2]):
                            zeta2 = basis_set[basis_fn2][3][prim2_idx]
                            coeff2 = basis_set[basis_fn2][4][prim2_idx]

                            p = zeta1 + zeta2
                            q = zeta1 * zeta2 / p
                            gaussian_center1 = (zeta1 * atom1_coords + zeta2 * atom2_coords) / p
                            factor12 = normalization_factor * np.exp(-q * distance12_squared) / p

                            for prim3_idx in range(basis_set[basis_fn3][2]):
                                zeta3 = basis_set[basis_fn3][3][prim3_idx]
                                coeff3 = basis_set[basis_fn3][4][prim3_idx]

                                for prim4_idx in range(basis_set[basis_fn4][2]):
                                    zeta4 = basis_set[basis_fn4][3][prim4_idx]
                                    coeff4 = basis_set[basis_fn4][4][prim4_idx]

                                    pk = zeta3 + zeta4
                                    qk = zeta3 * zeta4 / pk
                                    gaussian_center2 = (
                                        (zeta3 * atom3_coords + zeta4 * atom4_coords) / pk
                                    )
                                    factor34 = normalization_factor * np.exp(-qk * distance34_squared) / pk

                                    inter_center_distance_squared = np.linalg.norm(
                                        gaussian_center2 - gaussian_center1
                                    ) ** 2
                                    rho = p * pk / (p + pk)
                                    boys_factor = boys_function(rho * inter_center_distance_squared)

                                    integral_contribution = (
                                        boys_factor
                                        * factor12
                                        * factor34
                                        / np.sqrt(p + pk)
                                        * coeff1
                                        * coeff2
                                        * coeff3
                                        * coeff4
                                    )
                                    integral_value += integral_contribution

                    # Symmetry considerations
                    two_electron_integrals[basis_fn1, basis_fn2, basis_fn3, basis_fn4] = integral_value
                    two_electron_integrals[basis_fn2, basis_fn1, basis_fn3, basis_fn4] = integral_value
                    two_electron_integrals[basis_fn1, basis_fn2, basis_fn4, basis_fn3] = integral_value
                    two_electron_integrals[basis_fn2, basis_fn1, basis_fn4, basis_fn3] = integral_value
                    two_electron_integrals[basis_fn3, basis_fn4, basis_fn1, basis_fn2] = integral_value
                    two_electron_integrals[basis_fn4, basis_fn3, basis_fn1, basis_fn2] = integral_value
                    two_electron_integrals[basis_fn3, basis_fn4, basis_fn2, basis_fn1] = integral_value
                    two_electron_integrals[basis_fn4, basis_fn3, basis_fn2, basis_fn1] = integral_value

    return two_electron_integrals
