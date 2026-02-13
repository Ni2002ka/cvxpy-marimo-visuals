# marimo app: Huber objective with an M slider

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
    M_slider = mo.ui.slider(
        start=-5,
        stop=0,
        step=0.1,
        value=-5,
        label="Huber threshold M",
        show_value=True,
    )
    M_slider
    return (M_slider,)


@app.cell
def _(cp, np):
    # ---- Replace this cell with YOUR problem setup (u, cons) ----
    # If you already have u and cons elsewhere, you can delete this cell.

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


@app.cell
def _(M_slider, cons, cp, u):
    M_exp = float(M_slider.value)
    M = 10 ** M_exp

    h_obj = cp.Minimize(cp.sum(cp.huber(u, M)) / (2*M))
    h_prob = cp.Problem(h_obj, cons)
    h_prob.solve()

    u_val = None if u.value is None else u.value.copy()
    return M, h_prob, u_val


@app.cell
def _(M, h_prob, mo, u_val):
    mo.md(f"""
    **Status:** `{h_prob.status}`  
    **M:** `{M:g}`  
    **Optimal value:** `{h_prob.value:.6g}`  
    **u.value shape:** `{None if u_val is None else u_val.shape}`
    """)
    return


@app.cell
def _(M, np, plt, u_val):
    fig = plt.figure(figsize=(8, 5))

    plt.subplot(2, 1, 1)
    if u_val is not None:
        plt.plot(u_val.T)
    plt.ylabel(r"$u_t$")
    plt.title(rf"(e) $\sum \mathrm{{Huber}}_M(u)$, $M={M:g}$")
    plt.grid(True)
    plt.xlabel("t")

    plt.subplot(2, 1, 2)
    if u_val is not None:
        plt.plot(np.linalg.norm(u_val, axis=0), c="black")
    plt.ylabel(r"$\|u_t\|_2$")
    plt.grid(True)
    plt.xlabel("t")

    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
