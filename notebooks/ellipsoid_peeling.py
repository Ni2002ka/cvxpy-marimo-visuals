# marimo notebook: ellipsoid peeling (server-solved)
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
    return ChartPuck, SOLVE_URL, mo, np, sys


@app.cell
def _(mo):
    mo.md(r"""
    # Ellipsoid peeling using dual values
    ## Minimum volume ellipsoid around a set of points
    * Ellipsoid parametarized as $\varepsilon=\{\nu \mid \|A\nu + b\|_2 \le 1\}$.
    * Finite set of points $\{x_1, x_2, ..., x_m\}$.
    * Volume is proportional to $\det A^{-1}$.

    Server runs CVXPY+Clarabel and returns $(A,b)$ and duals; browser just renders.
    """)
    return


@app.cell
def _(ChartPuck, SOLVE_URL, mo, np, sys):
    import time
    import asyncio
    import json

    n = 5

    # ---- request helper ----
    async def post_positions(url: str, positions: np.ndarray):
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
            r = requests.post(url, json=payload, timeout=15)
            return r.json()

    # ---- cache ----
    cache = {
        "A": None,               # np.array (2,2)
        "b": None,               # np.array (2,)
        "lambdas": None,         # np.array (n,)
        "positions_used": None,  # np.array (n,2) positions used to produce A,b,lambdas
        "in_flight": False,
        "last_req_t": 0.0,
        "last_error": None,
        "last_ok_t": None,
        "last_sent_t": None,
    }

    # Debounce for server calls
    MIN_SECONDS_BETWEEN_SOLVES = 0.35

    async def request_solve(positions: np.ndarray):
        cache["in_flight"] = True
        cache["last_sent_t"] = time.monotonic()
        try:
            data = await post_positions(SOLVE_URL, positions)
            if data.get("ok"):
                cache["A"] = np.array(data["A"], dtype=float)
                cache["b"] = np.array(data["b"], dtype=float)
                cache["lambdas"] = np.array(data["lambdas"], dtype=float)
                cache["positions_used"] = positions.copy()
                cache["last_error"] = None
                cache["last_ok_t"] = time.monotonic()
            else:
                cache["last_error"] = data.get("error", "unknown error")
        except Exception as e:
            cache["last_error"] = f"{type(e).__name__}: {e}"
        finally:
            cache["in_flight"] = False

    def schedule(coro):
        """
        Robust task scheduling across:
        - Pyodide (running loop)
        - marimo run (running loop)
        - edge cases (no running loop) -> queue onto loop if available
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
            return True
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(coro)
                return True
            except Exception:
                return False

    def maybe_schedule_solve(positions: np.ndarray):
        now = time.monotonic()

        # If we have never solved successfully, be more aggressive
        min_gap = 0.05 if cache["A"] is None else MIN_SECONDS_BETWEEN_SOLVES

        if cache["in_flight"]:
            return
        if now - cache["last_req_t"] < min_gap:
            return

        cache["last_req_t"] = now
        ok = schedule(request_solve(positions))
        if not ok:
            cache["last_error"] = "Could not schedule async task (no event loop)"

    def draw_func(ax, widget):
        # current UI positions
        P_now = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        # kick a server solve periodically (debounced)
        maybe_schedule_solve(P_now)

        # draw current points always
        ax.scatter(P_now[:, 0], P_now[:, 1], s=200, label="points")

        Ahat = cache["A"]
        bhat = cache["b"]
        lambdas = cache["lambdas"]
        P_used = cache["positions_used"]

        # IMPORTANT: draw ellipse/arrows using P_used so they are consistent with the solve,
        # even if user is dragging and results are slightly stale.
        if Ahat is not None and bhat is not None and P_used is not None:
            theta = np.linspace(0, 2 * np.pi, 400)
            U = np.vstack([np.cos(theta), np.sin(theta)])
            try:
                X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))
                ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary")
            except np.linalg.LinAlgError:
                cache["last_error"] = "LinAlgError: singular A from server?"

            if lambdas is not None and len(lambdas) == n:
                lamb = np.maximum(lambdas, 0.0)
                Y = (Ahat @ P_used.T + bhat.reshape(2, 1))
                Ynorm = np.linalg.norm(Y, axis=0) + 1e-12
                N = (Ahat.T @ (Y / Ynorm))

                lam_scaled = lamb / np.max(lamb) if np.max(lamb) > 0 else lamb
                base = 2.0
                Ux = -N[0, :] * (base * lam_scaled)
                Uy = -N[1, :] * (base * lam_scaled)

                ax.quiver(
                    P_used[:, 0], P_used[:, 1],
                    Ux, Uy,
                    angles="xy", scale_units="xy", scale=1,
                    width=0.006, alpha=0.9,
                    label="dual normal force (scaled)",
                )

        # status/debug text
        status = []
        if cache["in_flight"]:
            status.append("solving...")
        else:
            status.append("idle")
        if cache["last_ok_t"] is not None:
            status.append(f"last ok: {time.monotonic() - cache['last_ok_t']:.2f}s ago")
        else:
            status.append("no successful solve yet")
        ax.text(-3.8, 3.65, " | ".join(status), fontsize=9)

        if cache["last_error"]:
            ax.text(-3.8, 3.40, f"solve error: {cache['last_error']}", fontsize=8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Drag points to move them (server-solved)")
        ax.grid(True, alpha=0.3)

    # Good init (non-degenerate)
    x0 = [-1.5, 1.5, 0.0, 0.0, 0.6]
    y0 = [ 1.5, 1.5, 0.0,-1.5,-0.2]

    multi_puck = ChartPuck.from_callback(
        draw_fn=draw_func,
        figsize=(8, 8),
        x_bounds=(-4, 4),
        y_bounds=(-4, 4),
        x=x0,
        y=y0,
        puck_color="red",
        throttle=60,
    )

    # Force an initial solve ASAP so it renders even before you drag
    P0 = np.array(list(zip(x0, y0)), dtype=float)
    schedule(request_solve(P0))

    multi_widget = mo.ui.anywidget(multi_puck)
    return (multi_widget,)


@app.cell
def _(multi_widget):
    multi_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes
    * Ellipse + arrows may lag slightly while dragging because solves are server-side.
    * They are always drawn consistently for the *positions used by the most recent solve*.
    """)
    return


if __name__ == "__main__":
    app.run()
