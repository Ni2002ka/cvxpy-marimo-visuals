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
    import matplotlib.pyplot as plt  # noqa: F401

    import sys
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("wigglystuff")

        import pyodide_js
        await pyodide_js.loadPackage("clarabel")
        import clarabel  # noqa: F401

    from wigglystuff import ChartPuck
    return ChartPuck, cp, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Ellipsoid peeling using dual values
    ## Minimum volume ellipsoid around a set of points
    * Ellipsoid parameterized as $\varepsilon=\{\nu \mid \|A\nu + b\|_2 \le 1\}$.
    * We solve (equivalently) $\min -\log\det(A)$ subject to $\|Ax_i+b\|_2 \le 1$.
    """)
    return


@app.cell
def _(mo):
    solve_btn = mo.ui.button(label="Solve ellipsoid")
    solve_btn
    return solve_btn


@app.cell
def _(ChartPuck, mo, np):
    n = 5

    # Cache for solution + duals (draw_func reads this, solve cell writes it)
    state = {"Ahat": None, "bhat": None, "lambdas": None}

    def draw_func(ax, widget):
        positions = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")

        Ahat = state["Ahat"]
        bhat = state["bhat"]
        lambdas = state["lambdas"]

        # Only draw ellipsoid/duals if we have a cached solution
        if Ahat is not None and bhat is not None:
            theta = np.linspace(0, 2*np.pi, 400)
            U = np.vstack([np.cos(theta), np.sin(theta)])
            X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))
            ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")

            if lambdas is not None:
                lambdas = np.maximum(np.asarray(lambdas, dtype=float), 0.0)

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
        ax.set_title("Drag points; click Solve to update")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    multi_puck = ChartPuck.from_callback(
        draw_fn=draw_func,
        figsize=(8, 8),
        x_bounds=(-4, 4),
        y_bounds=(-4, 4),
        x=[-1.5] + [0] * (n - 2) + [1.5],
        y=[0] + [1.5] * (n - 1),
        puck_color="red",
        throttle=100
    )

    multi_widget = mo.ui.anywidget(multi_puck)
    return multi_puck, multi_widget, n, state


@app.cell
def _(cp, multi_puck, multi_widget, n, np, solve_btn, state):
    # Run only when button is clicked
    _ = solve_btn.value

    positions = np.array([[multi_widget.x[i], multi_widget.y[i]] for i in range(n)], dtype=float)

    A = cp.Variable((2, 2), PSD=True)
    b = cp.Variable(2)

    cons = [cp.norm2(A @ positions[j] + b) <= 1 for j in range(n)]

    # For {x : ||A x + b|| <= 1}, volume ∝ det(A^{-1}), so minimize -log_det(A).
    # Small trace regularizer helps Clarabel converge in-browser.
    eps = 1e-2
    prob = cp.Problem(cp.Minimize(-cp.log_det(A) + eps * cp.trace(A)), cons)

    prob.solve(solver=cp.CLARABEL, max_iter=200, verbose=False)

    Ahat = A.value
    bhat = b.value
    if Ahat is None or bhat is None:
        raise ValueError(f"Solve failed: status={prob.status}")

    lambdas = np.array([c.dual_value for c in cons], dtype=float)

    state["Ahat"] = Ahat
    state["bhat"] = bhat
    state["lambdas"] = lambdas

    # Force redraw so ellipsoid appears
    multi_puck.redraw()
    return


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations:
    * Points not touching the boundary have dual ~0 (complementary slackness).
    * Interpreting the duals as “normal forces” gives intuition for which points are active.
    """)
    return


if __name__ == "__main__":
    app.run()
