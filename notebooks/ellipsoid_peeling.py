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
    import matplotlib.pyplot as plt

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
    * Ellipsoid parametarized as $\varepsilon=\{\nu \mid \|A\nu + b\|_2 \le 1\}$.
    * Finite set of points $\{x_1, x_2, ..., x_m\}$.
    * Volume is proportional to $\det(A^{-1})$.

    $$
    \min_{A,b}\ \log\det(A^{-1})
    \quad \text{s.t.}\quad \|Ax_i+b\|_2 \le 1,\ i=1,\dots,m
    $$
    """)
    return


@app.cell
def _(mo):
    # Button to trigger solving (prevents solve on initial widget construction)
    solve_btn = mo.ui.button(label="Solve ellipsoid")
    solve_btn
    return solve_btn


@app.cell
def _(ChartPuck, mo, np):
    n = 5

    # Shared cache for solution + duals (draw_func reads from this)
    state = {
        "Ahat": None,
        "bhat": None,
        "lambdas": None,
        "last_positions": None,
    }

    def draw_func(ax, widget):
        positions = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        ax.clear()
        ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")

        Ahat = state["Ahat"]
        bhat = state["bhat"]
        lambdas = state["lambdas"]

        # Only draw ellipsoid/duals if we have a cached solve
        # AND it corresponds to the current positions (optional but avoids confusion).
        if Ahat is not None and bhat is not None and lambdas is not None:
            last = state["last_positions"]
            if last is not None and last.shape == positions.shape and np.allclose(last, positions):
                theta = np.linspace(0, 2*np.pi, 400)
                U = np.vstack([np.cos(theta), np.sin(theta)])
                X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))
                ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")

                # Dual “normal forces”
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
            else:
                ax.set_title("Drag points, then click 'Solve ellipsoid'")

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
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

    # For this parameterization, minimize -log_det(A) (volume of A^{-1})
    eps = 1e-3
    prob = cp.Problem(cp.Minimize(-cp.log_det(A) + eps * cp.trace(A)), cons)

    prob.solve(
        solver=cp.CLARABEL,
        max_iter=200,
        verbose=False,
    )

    Ahat = A.value
    bhat = b.value

    if Ahat is None or bhat is None:
        raise ValueError(f"Solve failed: status={prob.status}")

    # Duals: one per constraint
    lambdas = np.array([c.dual_value for c in cons], dtype=float)
    lambdas = np.maximum(lambdas, 0.0)

    # Cache results for draw_func
    state["Ahat"] = Ahat
    state["bhat"] = bhat
    state["lambdas"] = lambdas
    state["last_positions"] = positions

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
    * Dual corresponding to each point demonstrates how much the volume of the ellipsoid would change if we removed that point.
    * If the ellipsoid were an elastic membrane, we can think of this as a "normal force" acting on each point.
    * Points not touching the boundary have zero "force" (their dual is ~0 by complementary slackness).
    """)
    return


if __name__ == "__main__":
    app.run()
