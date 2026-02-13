import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    return cp, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Control with various objectives

    * Optimal control problem, with dynamics $x_{t+1} = Ax_t + Bu_t$, with $t = 0,1,...,T − 1$.
      * $x_t \in \mathbf{R}^n$ is the state
      * $u_t \in \mathbf{R}^m$ is the control or input
      * $A \in \mathbf{R}^{n×n}$ is the dynamics matrix
      * $B \in \mathbf{R}^{n×m}$ is the input matrix.
    * Given the initial state: $x_0 = x_\text{init}$,
    * Require that the final state be zero: $x_T = 0$.

    ## Optimization Variables
    We want to find the sequence of inputs $u_0, ..., u_{T-1}$ that minimize a certain objective. We optimize over u's and x's, and apply the constraints that connect them to get the optimal u's.
    """)
    return


@app.cell
def _(cp, np):
    # Starter Code
    n = 4
    m = 2 

    A = np.array([
    [ 0.95,  0.16,  0.12,  0.01],
    [-0.12,  0.98, -0.11, -0.03],
    [-0.16,  0.02,  0.98,  0.03],
    [-0.  ,  0.02, -0.04,  1.03],
    ])

    B = np.array([
    [ 0.8 , 0. ],
    [ 0.1 , 0.2],
    [ 0.  , 0.8],
    [-0.2 , 0.1],
    ])

    x_init = np.ones(n)

    T = 100

    # Define CVXPY variables and constraints
    u = cp.Variable((m,T))
    x = cp.Variable((n,T+1))

    cons = [x[:,-1] == np.zeros(n)] # Final state is zero
    cons += [x[:,0] == x_init] # Initial state constraint

    for t in range(T):
       cons.append(x[:,t+1] == A@x[:,t] + B@u[:,t])
    return cons, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Different objectives to optimize for
    1. Sum of squares of 2-norms: $\sum_0^{T-1}||u_t||_2^2$
    3. Sum of 2-norms: $\sum_0^{T-1}||u_t||_2$
    4. Max of 2-norms: $\max_{[0,T-1]}||u_t||_2$
    5. Sum of 1-norms: $\sum_0^{T-1}||u_t||_1$
    """)
    return


@app.cell
def _(cons, cp, np, plt, u):
    import sys
    import matplotlib as mpl

    # Emscripten (html-wasm) cannot run subprocess, so Matplotlib usetex will crash.
    IS_WASM = (sys.platform == "emscripten")

    # Use real LaTeX locally; fall back to mathtext on Pages.
    mpl.rcParams["text.usetex"] = (not IS_WASM)

    objs = [
        (cp.Minimize(cp.sum_squares(u))              , r"(a) $\|u\|_2^2$"),
        (cp.Minimize(cp.sum(cp.norm(u, 2, axis=0)))  , r"(b) $\sum\|u_t\|_2$"),
        (cp.Minimize(cp.max(cp.norm(u, axis=0)))     , r"(c) $\max\|u_t\|_2$"),
        (cp.Minimize(cp.sum(cp.norm(u, 1, axis=0)))  , r"(d) $\sum\|u_t\|_1$"),
    ]

    fig = plt.figure(figsize=(15, 5))

    for i, (obj, label) in enumerate(objs):
        prob = cp.Problem(obj, cons)
        prob.solve()

        plt.subplot(2, 4, i + 1)
        plt.plot(u.value.T)
        if i == 0:
            plt.ylabel(r"$u_t$")
        plt.title(label)
        plt.grid()
        plt.xlabel("t")

        plt.subplot(2, 4, i + 5)
        plt.plot(np.linalg.norm(u.value, axis=0), label=r"$\|u_t\|_2$")
        if i == 2:
            plt.ylim(ymax=0.12, ymin=0)
        if i == 0:
            plt.ylabel(r"$\|u_t\|_2$")
        plt.grid()
        plt.xlabel("t")

    # tight_layout triggers text measurement; it's fine locally but can be flaky in wasm
    if not IS_WASM:
        plt.tight_layout()

    fig
    return


if __name__ == "__main__":
    app.run()
