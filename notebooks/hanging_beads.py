# marimo notebook: hanging rope with beads + adjustable endpoints + adjustable floor
# Run:  marimo run app.py
# (or open in editor and "marimo edit app.py")

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import matplotlib.pyplot as plt
    return cp, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Dual variables as contact forces

    **(Convex) Model:**
    - Beads sit at positions \(x_i, y_i\).
    - Endpoints are fixed.
    - Beads are connected by springs (with length-squared energy)
    - The overall energy is the sum of the gravitational potential energy and spring potential energy.

    \[
    E=\sum_{i=1}^n mgy_i + \frac{k}{2} \sum_{i=1}^{n-1} ((x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2)
    \]

    The objective is to minimize the total energy in the system, subject to
    - A horizontal **floor** at height $y_{\text{floor}}$: \(y \ge y_{\text{floor}}\). When active, beads rest on the floor.


    **Duals:**
    - For each bead, the dual variable on the floor constraint \(y_i \ge y_{\text{floor}}\) acts like a **normal-force multiplier**: it is nonzero exactly when the bead is “pushing into” the floor.
    """)
    return


@app.cell
def _(mo):
    # --- Controls ---
    # n_slider = mo.ui.slider(5, 31, value=11, step=2, label="Number of beads (odd is nice)")
    right_h = mo.ui.slider(-2.0, 2.0, value=2.0, step=0.1, label="Right endpoint height")
    # right_d = mo.ui.slider(0, 5.0, value=3.0, step=0.1, label="Right endpoint distance")
    # floor_h = mo.ui.slider(-3.0, 0, value=-2.0, step=0.1, label="Floor height")
    # seg_len = mo.ui.slider(0.2, 2.0, value=0.6, step=0.02, label="Max segment length L")
    k_coeff = mo.ui.slider(0.1, 100.0, value=47.0, step=0.1, label="String tension coefficient")
    mass = mo.ui.slider(0.1, 10.0, value=1.0, step=0.1, label="Each bead's mass")


    mo.hstack(
        [
            # mo.vstack([seg_len]),
            mo.vstack([k_coeff, right_h])
        ],
        justify="space-between",
    )
    return k_coeff, right_h


@app.cell
def _(k_coeff, right_h):

    # n = int(n_slider.value)
    n= 11
    m = 1.0
    k = float(k_coeff.value)
    # L = float(seg_len.value)
    g = 9.8
    yR = float(right_h.value)
    xR = 4.0
    # y_floor = float(floor_h.value)
    y_floor = -2.0
    return g, k, m, n, xR, yR, y_floor


@app.cell
def _(cp, g, k, m, n, np, xR, yR, y_floor):

    y = cp.Variable(n)
    x = cp.Variable(n)

    constraints = []

    # Endpoint fixed
    constraints.append(y[0] == 0)
    constraints.append(x[0] == 0)
    constraints.append(y[-1] == yR)
    constraints.append(x[-1] == xR)

    # Floor constraints: y_i >= y_floor
    floor_cons = (y >= y_floor)
    constraints.append(floor_cons)

    E = m * g * cp.sum(y)
    for i in range(n - 1):
        # Vector diff to select two consequetive entries
        diff = np.zeros(n)
        diff[i+1] = 1
        diff[i] = -1
        l_segment_squared = cp.square(diff @ x) + cp.square(diff @ y)
        E += 0.5 * k * l_segment_squared
        # constraints += [l_segment_squared <= L**2]


    # Potential energy (up to constant): minimize sum m_i g y_i
    obj = cp.Minimize(E)

    prob = cp.Problem(obj, constraints)

    # Solve
    try:
        min_energy = prob.solve()
        # print("Minimum energy in the system: ", min_energy)
    except Exception:
        print("solve error!")


    y_opt = y.value.astype(float)
    x_opt = x.value.astype(float)

    # Duals give normal forces
    normal_forces =  floor_cons.dual_value
    normal_forces = np.maximum(0.0, normal_forces)

    # Dual forces at endpoints:
    left_endpt_y = constraints[0].dual_value
    left_endpt_x = constraints[1].dual_value
    right_endpt_y = constraints[2].dual_value
    right_endpt_x = constraints[3].dual_value
    return (
        left_endpt_x,
        left_endpt_y,
        normal_forces,
        right_endpt_x,
        right_endpt_y,
        x_opt,
        y_opt,
    )


@app.cell
def _(
    left_endpt_x,
    left_endpt_y,
    normal_forces,
    np,
    plt,
    right_endpt_x,
    right_endpt_y,
    xR,
    x_opt,
    yR,
    y_floor,
    y_opt,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    # Rope
    ax.plot(
        x_opt, y_opt,
        color="C0",
        linewidth=2,
        zorder=2,
        label="rope"
    )

    # Beads
    ax.scatter(
        x_opt, y_opt,
        s=40,
        color="C0",
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="beads"
    )

    # Endpoints
    ax.scatter(
        [x_opt[0], x_opt[-1]],
        [y_opt[0], y_opt[-1]],
        s=80,
        color="C3",
        zorder=4,
        label="fixed endpoints"
    )

    # Floor
    ax.axhline(
        y_floor,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
        label="floor"
    )

    # Formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimal hanging rope configuration")

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    # # Parameter box
    # param_text = (
    #     rf"$n={n}$" "\n"
    #     rf"$g={g:.2f}$" "\n"
    #     rf"$y_{{\mathrm{{floor}}}}={y_floor:.2f}$"
    # )

    # ax.text(
    #     0.02, 0.98,
    #     # param_text,
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     fontsize=10,
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    # )


    # Only show meaningful forces
    mask = normal_forces > 1e-6

    if np.any(mask):
        # Scale arrows nicely relative to plot height
        y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        scale = 0.15 * y_span / normal_forces[mask].max()

        for xi, Ni in zip(x_opt[mask], normal_forces[mask]):
            ax.arrow(
                xi,
                y_floor,
                0.0,
                scale * Ni,
                width=0.005,
                head_width=0.04,
                head_length=0.05,
                length_includes_head=True,
                color="C2",
                alpha=0.85,
                zorder=5,
            )

        # Legend handle
        ax.plot([], [], color="C2", linewidth=2, label="normal force")


    # Endpoint forces arrows
    ax.arrow(0, 0, -0.01 * left_endpt_x, -0.01 * left_endpt_y,         
                width=0.005,
                head_width=0.04,
                head_length=0.05,
                length_includes_head=True,
                color="C6",
                alpha=0.85,
                zorder=5,)

    ax.arrow(xR, yR, -0.01 * right_endpt_x, -0.01 * right_endpt_y,         
                width=0.005,
                head_width=0.04,
                head_length=0.05,
                length_includes_head=True,
                color="C6",
                alpha=0.85,
                zorder=5,)

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2.5, 2.8)
    ax.set_aspect('equal')

    plt.tight_layout()
    # plt.show()
    fig
    return


if __name__ == "__main__":
    app.run()
