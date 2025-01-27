import argparse
from scf import scf
from utils import read_input, nuclear_repulsion_energy
from visualization import (
    plot_scf_convergence,
    plot_geometry,
    plot_molecular_orbital,
)
from integrals import build_basis, compute_one_electron_integrals, compute_two_electron_integrals


def run_calculation(input_file, basis_set_name, basis_dir="basis/"):
    natoms, charge, labels, z, coords = read_input(input_file)
    nb, basis = build_basis(natoms, z, coords, basis_set_name, basis_dir)
    S, T, V = compute_one_electron_integrals(natoms, nb, z, coords, basis)
    TEI = compute_two_electron_integrals(nb, coords, basis)
    E_elec, P, C, energies = scf(T, V, S, TEI, nb, z, charge)
    E_nuc = nuclear_repulsion_energy(z, coords)
    return natoms, charge, labels, z, coords, nb, basis, C, E_elec, E_nuc, energies


def interactive_mode():
    while True:
        input_file = input("Enter the path to the input file: ").strip()
        basis_set_name = input("Enter the basis set name: ").strip()
        try:
            data = run_calculation(input_file, basis_set_name)
            natoms, charge, labels, z, coords, nb, basis, C, E_elec, E_nuc, energies = data
            print(f"Electronic energy: {E_elec:.6f} a.u.")
            print(f"Nuclear repulsion energy: {E_nuc:.6f} a.u.")
            print(f"Total Hartree-Fock energy: {E_elec + E_nuc:.6f} a.u.")
        except Exception as e:
            print(f"Error: {e}")
            continue

        while True:
            print("1. SCF Energy Convergence")
            print("2. Molecular Geometry")
            print("3. Molecular Orbitals")
            print("4. Change Input/Basis")
            print("5. Exit Program")
            choice = input("Enter your choice: ").strip()

            if choice == "1":
                plot_scf_convergence(energies)
            elif choice == "2":
                plot_geometry(coords, labels)
            elif choice == "3":
                plot_molecular_orbital(coords, labels, C, basis)
            elif choice == "4":
                break
            elif choice == "5":
                return
            else:
                print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to the input file.")
    parser.add_argument("-b", "--basis", help="Name of the basis set.")
    args = parser.parse_args()

    if args.input and args.basis:
        try:
            data = run_calculation(args.input, args.basis)
            natoms, charge, labels, z, coords, nb, basis, C, E_elec, E_nuc, energies = data
            print(f"Electronic energy: {E_elec:.6f} a.u.")
            print(f"Nuclear repulsion energy: {E_nuc:.6f} a.u.")
            print(f"Total Hartree-Fock energy: {E_elec + E_nuc:.6f} a.u.")
            print("Visualizations available: SCF, Geometry, MO.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
