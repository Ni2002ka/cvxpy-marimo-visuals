# marimo notebook: ellipsoid peeling (server-solved, debounced + latest-wins)
# Run: marimo run app.py

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
    # Ellipsoid peeling using dual values (server-solved)
    Smoother UX: debounce solves + discard stale responses.
    """)
    return


@app.cell
def _(ChartPuck, SOLVE_URL, mo, np, sys):
    import time
    import asyncio
    import json

    n = 5

    # --------- tuning knobs ----------
    DEBOUNCE_SECONDS = 0.45     # only solve after user stops moving for this long
    MIN_GAP_SECONDS = 0.20      # never solve more frequently than this
    # --------------------------------

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

    def schedule(coro):
        try:
            asyncio.get_running_loop().create_task(coro)
            return True
        except RuntimeError:
            try:
                asyncio.get_event_loop().create_task(coro)
                return True
            except Exception:
                return False

    cache = {
        "A": None,
        "b": None,
        "lambdas": None,
        "positions_used": None,

        "last_error": None,

        # debounce state
        "last_positions": None,
        "last_move_t": 0.0,
        "last_solve_start_t": 0.0,

        # latest-wins state
        "in_flight": False,
        "req_id": 0,          # increment for each request
        "latest_req_id": 0,   # id of latest request we care about

        "last_ok_t": None,
    }

    async def request_solve(positions: np.ndarray, req_id: int):
        cache["in_flight"] = True
        try:
            data = await post_positions(SOLVE_URL, positions)

            # Discard stale responses
            if req_id != cache["latest_req_id"]:
                return

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
            # Only clear in_flight if we're still the latest request
            if req_id == cache["latest_req_id"]:
                cache["in_flight"] = False

    def maybe_debounced_solve(P_now: np.ndarray):
        now = time.monotonic()

        # track movement (treat as moved if any coordinate changed)
        if cache["last_positions"] is None:
            cache["last_positions"] = P_now.copy()
            cache["last_move_t"] = now
        else:
            if np.max(np.abs(cache["last_positions"] - P_now)) > 1e-6:
                cache["last_positions"] = P_now.copy()
                cache["last_move_t"] = now

        # wait until user pauses
        paused_long_enough = (now - cache["last_move_t"]) >= DEBOUNCE_SECONDS
        if not paused_long_enough:
            return

        # throttle
        if now - cache["last_solve_start_t"] < MIN_GAP_SECONDS:
            return

        # start a new request (latest-wins)
        cache["last_solve_start_t"] = now
        cache["req_id"] += 1
        cache["latest_req_id"] = cache["req_id"]
        cache["in_flight"] = True
        schedule(request_solve(P_now.copy(), cache["latest_req_id"]))

    def draw_func(ax, widget):
        P_now = np.array([[widget.x[i], widget.y[i]] for i in range(n)], dtype=float)

        # debounce scheduling (solve only when user pauses)
        maybe_debounced_solve(P_now)

        # draw current points always
        ax.scatter(P_now[:, 0], P_now[:, 1], s=200, label="points")

        # draw last solved ellipse (may be stale while dragging)
        Ahat = cache["A"]
        bhat = cache["b"]
        lambdas = cache["lambdas"]
        P_used = cache["positions_used"]

        if Ahat is not None and bhat is not None and P_used is not None:
            theta = np.linspace(0, 2*np.pi, 400)
            U = np.vstack([np.cos(theta), np.sin(theta)])
            try:
                X = np.linalg.solve(Ahat, U - bhat.reshape(2, 1))
                ax.plot(X[0, :], X[1, :], linewidth=2, label="ellipsoid boundary (last solve)")
            except np.linalg.LinAlgError:
                pass

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
                    label="dual normal force (last solve)",
                )

        # status line
        msg = []
        if cache["in_flight"]:
            msg.append("solving (server)…")
        else:
            msg.append("idle")
        if cache["last_ok_t"] is None:
            msg.append("no solve yet")
        else:
            msg.append(f"last solve {time.monotonic() - cache['last_ok_t']:.1f}s ago")
        ax.text(-3.8, 3.65, " | ".join(msg), fontsize=9)

        if cache["last_error"]:
            ax.text(-3.8, 3.40, f"solve error: {cache['last_error']}", fontsize=8)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Drag points to move them (server-solved, debounced)")
        ax.grid(True, alpha=0.3)

    # init points
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
        throttle=30,  # redraw for smooth dragging; solve is debounced anyway
    )

    # kick one initial solve
    P0 = np.array(list(zip(x0, y0)), dtype=float)
    cache["req_id"] += 1
    cache["latest_req_id"] = cache["req_id"]
    cache["in_flight"] = True
    schedule(request_solve(P0, cache["latest_req_id"]))

    multi_widget = mo.ui.anywidget(multi_puck)
    return (multi_widget,)


@app.cell
def _(multi_widget):
    multi_widget
    return


if __name__ == "__main__":
    app.run()
