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


    def draw_func(ax, widget):
        ax.set_title("hello")
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
