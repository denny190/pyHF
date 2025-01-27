import os
import argparse
from scf import scf
from utils import read_input, nuclear_repulsion_energy
from visualization import plot_scf_and_geometry
from integrals import build_basis, compute_one_electron_integrals, compute_two_electron_integrals


def list_available_basis_sets(basis_dir):
    basis_files = [f for f in os.listdir(basis_dir) if f.endswith(".json")]
    return [os.path.splitext(f)[0] for f in basis_files]


def main():
    # Command-line interface
    parser = argparse.ArgumentParser(description="Run a Hartree-Fock SCF calculation.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input file (e.g., inputs/h2.input).")
    parser.add_argument("-b", "--basis", required=True, help="Name of the basis set (e.g., 6-31g).")
    parser.add_argument("--basis-dir", default="basis/", help="Path to the directory containing basis set JSON files.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    available_basis_sets = list_available_basis_sets(args.basis_dir)
    if args.basis not in available_basis_sets:
        print(f"Error: Basis set '{args.basis}' not found in directory '{args.basis_dir}'.")
        print("Available basis sets:")
        for basis in available_basis_sets:
            print(f"  - {basis}")
        return

    input_file = args.input
    basis_set_name = args.basis
    basis_dir = args.basis_dir

    print(f"Using input file: {input_file}")
    print(f"Using basis set: {basis_set_name} from directory '{basis_dir}'")

    natoms, charge, labels, z, coords = read_input(input_file)
    print(f"Read input: {natoms} atoms, charge = {charge}")

    nb, basis = build_basis(natoms, z, coords, basis_set_name, basis_dir)#
    print(f"Basis set: {nb} basis functions.")

    S, T, V = compute_one_electron_integrals(natoms, nb, z, coords, basis)
    TEI = compute_two_electron_integrals(nb, coords, basis)

    print(f"Type of nb before SCF: {type(nb)}") 
    E_elec, P, C, energies = scf(T, V, S, TEI, nb, z, charge)
    print(f"SCF completed. Electronic energy = {E_elec}")

    E_nuc = nuclear_repulsion_energy(z, coords)
    print(f"Nuclear repulsion energy = {E_nuc}")
    print(f"Total energy (HF) = {E_elec + E_nuc}")

    print("Generating visualizations...")
    plot_scf_and_geometry(
    coords, labels, C, basis, grid_size=50, extent=5.0, energies=energies
    )
    print("Calculation and visualization complete.")


if __name__ == "__main__":
    main()
