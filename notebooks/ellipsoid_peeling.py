# marimo notebook: ellipsoid peeling with convex hull
# Run:  marimo run app.py

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("wigglystuff")

    from wigglystuff import ChartPuck

    SOLVE_URL = "https://cvxpy-marimo-visuals.onrender.com/solve"
    return ChartPuck, SOLVE_URL, mo, np


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
def _(ChartPuck, SOLVE_URL, mo, np):
    import time
    import asyncio

    n = 5

    # ---- request helper ----
    async def post_positions(url: str, positions: np.ndarray):
        import json
        payload = {"positions": positions.tolist(), "max_iter": 50}

        if sys.platform == "emscripten":
            from pyodide.http import pyfetch
            resp = await pyfetch(
                url,
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps(payload),
            )
            return await resp.json()
        else:
            import requests
            r = requests.post(url, json=payload, timeout=10)
            return r.json()

    # ---- cache + debounce ----
    cache = {
        "A": None,         # np.array (2,2)
        "b": None,         # np.array (2,)
        "lambdas": None,   # np.array (n,)
        "in_flight": False,
        "last_req_t": 0.0,
        "last_error": None,
    }
    MIN_SECONDS_BETWEEN_SOLVES = 0.35

    async def request_solve(positions: np.ndarray):
        if cache["in_flight"]:
            return
        cache["in_flight"] = True
        try:
            data = await post_positions(SOLVE_URL, positions)
            if data.get("ok"):
                cache["A"] = np.array(data["A"], dtype=float)
                cache["b"] = np.array(data["b"], dtype=float)
                cache["lambdas"] = np.array(data["lambdas"], dtype=float)
                cache["last_error"] = None
            else:
                cache["last_error"] = data.get("error", "unknown error")
        except Exception as e:
            cache["last_error"] = f"{type(e).__name__}: {e}"
        finally:
            cache["in_flight"] = False

    def maybe_schedule_solve(positions: np.ndarray):
        now = time.monotonic()
        if now - cache["last_req_t"] < MIN_SECONDS_BETWEEN_SOLVES:
            return
        cache["last_req_t"] = now
        try:
            asyncio.create_task(request_solve(positions))
        except RuntimeError:
            # No event loop available; ignore (rare in marimo)
            pass

    def draw_func(ax, widget):
        positions = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        # schedule server solve (debounced)
        maybe_schedule_solve(positions)

        # draw points always
        ax.scatter(positions[:, 0], positions[:, 1], s=20, label="points")

        Ahat = cache["A"]
        bhat = cache["b"]
        lambdas = cache["lambdas"]

        if Ahat is not None and bhat is not None:
            # boundary
            theta = np.linspace(0, 2*np.pi, 400)
            U = np.vstack([np.cos(theta), np.sin(theta)])
            X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))
            ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")

            # dual arrows
            if lambdas is not None and len(lambdas) == n:
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

        if cache["last_error"]:
            ax.text(-3.8, 3.6, f"solve error: {cache['last_error']}", fontsize=8)

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
        y=[1.5] + [-1.5] * (n-2) + [1.5],
        puck_color="red",
        throttle=60,  # redraw often; server solve is debounced
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
