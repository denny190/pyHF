from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# Define atom colors based on element type
ATOM_COLORS = {
    "H": "white",
    "He": "cyan",
    "C": "gray",
    "O": "red",
    "N": "blue",
    "S": "yellow",
    # Add more elements as needed
}

def plot_scf_convergence(energies):
    """
    Plot the SCF energy convergence in a separate window.

    Args:
        energies (list): List of electronic energies from SCF iterations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(energies) + 1), energies, marker="o", linestyle="-")
    ax.set_title("SCF Energy Convergence", fontsize=16)
    ax.set_xlabel("Iteration Number", fontsize=14)
    ax.set_ylabel("Electronic Energy (a.u.)", fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_geometry(coords, labels):
    """
    Plot the molecular geometry in 3D in a separate window.

    Args:
        coords (np.ndarray): Atomic coordinates (natoms x 3).
        labels (list): Atomic labels (e.g., ["H", "O"]).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot atoms
    seen_labels = set()
    for i, coord in enumerate(coords):
        color = ATOM_COLORS.get(labels[i], "black")
        label = labels[i] if labels[i] not in seen_labels else None
        seen_labels.add(labels[i])
        ax.scatter(*coord, s=300, color=color, edgecolor="k", label=label)

    ax.set_title("Molecular Geometry", fontsize=16)
    ax.set_xlabel("X (Bohr)", fontsize=14)
    ax.set_ylabel("Y (Bohr)", fontsize=14)
    ax.set_zlabel("Z (Bohr)", fontsize=14)
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_molecular_orbital(coords, labels, C, basis, grid_size=50, extent=5.0):
    num_mos = C.shape[1]
    current_mo = [0]
    colorbar = None

    fig = plt.figure(figsize=(12, 6))  # Increased figure size to accommodate sliders
    ax = fig.add_subplot(111, projection="3d")

    # Offset sliders further to the right
    slider_transparency_ax = plt.axes([0.9, 0.2, 0.03, 0.6], facecolor="lightgoldenrodyellow")
    transparency_slider = Slider(slider_transparency_ax, "Transparency", 0.1, 1.0, valinit=0.5, orientation="vertical")

    slider_mo_ax = plt.axes([0.91, 0.2, 0.03, 0.6], facecolor="lightgoldenrodyellow")
    mo_slider = Slider(slider_mo_ax, "MO Index", 1, num_mos, valinit=1, valstep=1, orientation="vertical")

    def plot_geometry_and_mo(mo_index, transparency):
        nonlocal colorbar
        ax.clear()
        ax.set_title(f"Molecular Orbital {mo_index + 1}", fontsize=16)
        ax.set_xlabel("X (Bohr)", fontsize=14)
        ax.set_ylabel("Y (Bohr)", fontsize=14)
        ax.set_zlabel("Z (Bohr)", fontsize=14)

        X, Y, Z, MO_values = evaluate_mo_on_grid(coords, C, basis, grid_size, extent, mo_index)

        seen_labels = set()
        for i, coord in enumerate(coords):
            color = ATOM_COLORS.get(labels[i], "black")
            label = labels[i] if labels[i] not in seen_labels else None
            seen_labels.add(labels[i])
            ax.scatter(*coord, s=300, color=color, edgecolor="k", label=label)

        mask = np.abs(MO_values) > 0.01
        x_points = X[mask]
        y_points = Y[mask]
        z_points = Z[mask]
        mo_values_filtered = MO_values[mask]

        scatter = ax.scatter(
            x_points,
            y_points,
            z_points,
            c=mo_values_filtered,
            cmap="seismic",
            alpha=transparency,
            s=10,
        )

        if colorbar:
            colorbar.remove()

        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        colorbar.set_label("MO Value", fontsize=12)

        ax.legend(loc="upper right", fontsize=12)
        ax.view_init(elev=30, azim=45)

    plot_geometry_and_mo(mo_index=current_mo[0], transparency=0.5)

    def update_transparency(val):
        transparency = transparency_slider.val
        plot_geometry_and_mo(mo_index=current_mo[0], transparency=transparency)
        fig.canvas.draw_idle()

    def update_mo(val):
        current_mo[0] = int(mo_slider.val) - 1
        transparency = transparency_slider.val
        plot_geometry_and_mo(mo_index=current_mo[0], transparency=transparency)
        fig.canvas.draw_idle()

    transparency_slider.on_changed(update_transparency)
    mo_slider.on_changed(update_mo)

    plt.tight_layout()
    plt.show()


def evaluate_mo_on_grid(coords, C, basis, grid_size=50, extent=5.0, mo_index=0):
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    z = np.linspace(-extent, extent, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)
    MO_values = np.zeros_like(X)

    for i, basis_fn in enumerate(basis):
        atom_idx = basis_fn[1]
        atom_coords = coords[atom_idx]
        exponents = basis_fn[3]
        coefficients = basis_fn[4]
        angular_momentum = basis_fn[5]

        for zeta, coeff in zip(exponents, coefficients):
            r2 = (X - atom_coords[0]) ** 2 + (Y - atom_coords[1]) ** 2 + (Z - atom_coords[2]) ** 2
            primitive = coeff * np.exp(-zeta * r2)

            if angular_momentum == 0:
                MO_values += C[i, mo_index] * primitive

    return X, Y, Z, MO_values
