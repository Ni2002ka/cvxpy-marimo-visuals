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
        
        import pyodide_js
        await pyodide_js.loadPackage("clarabel")  # make clarabel available in-browser
        import clarabel

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
    n = 5

    # --- Build problem ONCE ---
    positions_param = cp.Parameter((n, 2))
    A = cp.Variable((2, 2), PSD=True)
    b = cp.Variable(2)

    cons = [cp.norm2(A @ positions_param[j, :] + b) <= 1 for j in range(n)]
    prob = cp.Problem(cp.Maximize(cp.log_det(A)), cons)

    # Cache last successful solution so we can still draw if solve fails
    last = {"A": None, "b": None}

    def draw_func(ax, widget):
        positions = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        positions_param.value = positions

        try:
            prob.solve(
                solver=cp.CLARABEL,
                warm_start=True,     # <-- important
                max_iter=30,         # lower in WASM to keep UI responsive
                verbose=False,
            )
        except Exception:
            pass

        Ahat = A.value if A.value is not None else last["A"]
        bhat = b.value if b.value is not None else last["b"]

        if Ahat is None or bhat is None:
            # If first solve hasn't succeeded yet, just draw points and return
            ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)
            return

        last["A"], last["b"] = Ahat, bhat

        # parameterize boundary
        theta = np.linspace(0, 2*np.pi, 400)
        U = np.vstack([np.cos(theta), np.sin(theta)])
        X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))

        ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")
        ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")

        # Duals
        lambdas = np.array([c.dual_value for c in cons], dtype=float)
        lambdas = np.maximum(lambdas, 0.0)

        Y = (Ahat @ positions.T + bhat.reshape(2, 1))
        Ynorm = np.linalg.norm(Y, axis=0) + 1e-12
        N = (Ahat.T @ (Y / Ynorm))

        lam_scaled = lambdas / np.max(lambdas) if np.max(lambdas) > 0 else lambdas
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
        throttle=250,   # <-- bump this in WASM; 100ms is still too chatty
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
