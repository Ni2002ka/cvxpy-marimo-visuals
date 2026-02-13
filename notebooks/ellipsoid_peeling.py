# marimo notebook: ellipsoid peeling with convex hull
# Run:  marimo run app.py

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import matplotlib.pyplot as plt

    import sys
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("wigglystuff")
        await micropip.install("scs")

    from wigglystuff import ChartPuck
    return ChartPuck, cp, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Ellipsoid peeling using dual values
    ## Minimum volume ellipsoid around a set of points
    * Ellipsoid parametarized as $\varepsilon=\{\nu | ||A\nu +b||_2 \leq 1\}$.
    * Finite set of points $\{x_1, x_2, ..., x_m\}$.
    * Volume is proportional to $\det A^{-1}$, so can find $\varepsilon$ by solving the convex problem

    $$
    \text{minimize (over } A,b) \;\;\;\; \log\det A^{-1} \\
    \text{subject to } ||Ax_i+b||_2 \leq 1, \;\;\;\; i=1,..., m
    $$
    """)
    return


@app.cell
def _(ChartPuck, cp, mo, np):
    n = 7


    def draw_func(ax, widget):
        positions = np.array([[widget.x[i], widget.y[i]] for i in range(n)])
        A = cp.Variable((2, 2), PSD=True)
        b = cp.Variable(2)

        cons = []
        for j in range(n):
            cons += [cp.norm2(A @ positions[j] + b) <= 1]

        prob = cp.Problem(cp.Maximize(cp.log_det(A)), cons)
        prob.solve(solver=cp.SCS)


        Ahat = A.value
        bhat = b.value

        if Ahat is None or bhat is None:
            raise ValueError("No values set for A or b")

        # parameterize boundary
        theta = np.linspace(0, 2*np.pi, 400)
        U = np.vstack([np.cos(theta), np.sin(theta)])          # Pick points on unit circle
        X = np.linalg.solve(Ahat, U - bhat.reshape(2,1))       # Find pre-image of points


        ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")
        ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")

        # Duals calculation
        lambdas = np.array([c.dual_value for c in cons], dtype=float)  # shape (n,)
        lambdas = np.maximum(lambdas, 0.0)

        Y = (Ahat @ positions.T + bhat.reshape(2, 1))
        Ynorm = np.linalg.norm(Y, axis=0) + 1e-12
        N = (Ahat.T @ (Y / Ynorm))

        if np.max(lambdas) > 0:
            lam_scaled = lambdas / np.max(lambdas)
        else:
            lam_scaled = lambdas

        base = 2.0
        Ux = -N[0, :] * (base * lam_scaled)
        Uy = -N[1, :] * (base * lam_scaled)

        ax.quiver(
            positions[:, 0], positions[:, 1],
            Ux, Uy,
            angles="xy", scale_units="xy", scale=1,
            width=0.006, alpha=0.9,
            label="dual normal force (scaled)"
        )

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Drag points to move them")
        ax.grid(True, alpha=0.3)

    multi_puck = ChartPuck.from_callback(
        draw_fn=draw_func,
        figsize=(8, 8),
        x_bounds=(-4, 4),
        y_bounds=(-4, 4),
        x=[-1.5] + [0] * (n-2) + [1.5],
        y=[0] + [1.5] * (n-1),
        puck_color="red",
        throttle=100
    )

    multi_widget = mo.ui.anywidget(multi_puck)
    return (multi_widget,)


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations:
    * Dual corresponding to each point demonstrates how much the volume of the elliposid would shrink if we were to remove that point.
    * If the ellipsoid were an ellastic membrane, we could think of this as a "normal force" acting on each point.
    * The points not touching the boundary have no "force" exerted on them. Their corresponding dual is zero (complementary slackness)
    """)
    return


if __name__ == "__main__":
    app.run()
