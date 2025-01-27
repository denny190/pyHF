from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

ATOM_COLORS = {
    "H": "white",
    "He": "cyan",
    "C": "gray",
    "O": "red",
    "N": "blue",
    "S": "yellow",
    # more elems as needed
}


def plot_scf_and_geometry(coords, labels, C, basis, grid_size=50, extent=5.0, energies=None):
    num_mos = C.shape[1]

    fig = plt.figure(figsize=(18, 8))
    ax_scf = fig.add_subplot(121)
    ax_geometry = fig.add_subplot(122, projection="3d")

    if energies is not None:
        ax_scf.plot(range(1, len(energies) + 1), energies, marker="o", linestyle="-")
        ax_scf.set_title("SCF Energy Convergence", fontsize=16)
        ax_scf.set_xlabel("Iteration Number", fontsize=14)
        ax_scf.set_ylabel("Electronic Energy (a.u.)", fontsize=14)
        ax_scf.grid(True)

    toolbar_ax = plt.axes([0.85, 0.1, 0.1, 0.8], frameon=True, facecolor="lightgray")
    toolbar_ax.axis("off") 

    slider_transparency_ax = plt.axes([0.86, 0.6, 0.02, 0.25], facecolor="lightgoldenrodyellow")
    transparency_slider = Slider(
        slider_transparency_ax, "Transparency", 0.1, 1.0, valinit=0.5, orientation="vertical"
    )

    slider_mo_ax = plt.axes([0.86, 0.25, 0.02, 0.25], facecolor="lightgoldenrodyellow")
    mo_slider = Slider(
        slider_mo_ax, "MO Index", 1, num_mos, valinit=1, valstep=1, orientation="vertical"
    )

    current_mo = [0] 
    colorbar = None  

    def plot_geometry_and_mo(mo_index, transparency):
        nonlocal colorbar
        ax_geometry.clear()
        ax_geometry.set_title(f"Molecular Geometry + MO {mo_index + 1}", fontsize=16)
        ax_geometry.set_xlabel("X (Bohr)", fontsize=14)
        ax_geometry.set_ylabel("Y (Bohr)", fontsize=14)
        ax_geometry.set_zlabel("Z (Bohr)", fontsize=14)

        X, Y, Z, MO_values = evaluate_mo_on_grid(coords, C, basis, grid_size, extent, mo_index)

        seen_labels = set()
        for i, coord in enumerate(coords):
            color = ATOM_COLORS.get(labels[i], "black")
            label = labels[i] if labels[i] not in seen_labels else None
            seen_labels.add(labels[i])
            ax_geometry.scatter(*coord, s=300, color=color, edgecolor="k", label=label)

        mask = np.abs(MO_values) > 0.01  # Threshold to avoid clutter
        x_points = X[mask]
        y_points = Y[mask]
        z_points = Z[mask]
        mo_values_filtered = MO_values[mask]

        scatter = ax_geometry.scatter(
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

        colorbar = fig.colorbar(scatter, ax=ax_geometry, shrink=0.6, pad=0.1)
        colorbar.set_label("MO Value", fontsize=12)

        ax_geometry.legend(loc="upper right", fontsize=12)
        ax_geometry.view_init(elev=30, azim=45)

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
            r2 = (X - atom_coords[0])**2 + (Y - atom_coords[1])**2 + (Z - atom_coords[2])**2
            primitive = coeff * np.exp(-zeta * r2)

            # Only 's' orbitals considered for simplicity
            if angular_momentum == 0:
                MO_values += C[i, mo_index] * primitive

    return X, Y, Z, MO_values
