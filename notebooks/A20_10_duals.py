import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Energy storage trade-offs

    * $t$: Time period, ranging from $1$ to $T$
    * $p_t$: (positive, time-varying) electricity price
    * $u_t$: (nonnegative) usage or consumption
    * $q_t$: (nonnegative) energy stored in the battery in period $t$
    * $c_t$: charging of the battery in period $t$
      > $c_t>0$: battery is charged
      >
      > $c_t<0$: battery is discharged

    * We neglect energy loss. So energy conservation means that $q_{t+1}=q_t+c_t$, for $1 \leq t \leq T-1$.

    * We finish with the same battery charge as when we started with so $q_1=q_T+c_T$
    * With the battery, the net consumption in period $t$ is $u_t+c_t$. We are not allowed to pump power back into the grid, so we require this to be nonnegative. i.e
      > $u_t+c_t \geq 0$
    * Total cost is $p^T(u+c)$
    * Battery capacity $Q$: $q_t\leq Q$
    * Maximum battery charge rate $C$, and max discharge rate $D$, so $-D\leq c_t\leq C$.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt


    # Problem data
    np.random.seed(1)

    T = 96
    t = np.linspace(1, T, num=T).reshape(T,1)
    p = np.exp(-np.cos((t-15)*2*np.pi/T)+0.01*np.random.randn(T,1))
    u = 2*np.exp(-0.6*np.cos((t+40)*np.pi/T) \
    - 0.7*np.cos(t*4*np.pi/T)+0.01*np.random.randn(T,1))
    plt.figure(1)
    plt.plot(t/4, p)
    plt.plot(t/4, u, "r")

    plt.legend(['p', 'u'])
    plt.show()
    return T, cp, np, p, plt, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Objective: Minimize Total Cost
    minimize $p^T(u+c)$

    ## Constraints
    * Charge/discharge limits $-D\mathbf{1}\leq c\leq C\mathbf{1}$
    * No pumping power back into the grid: $u+c\geq 0$
    * Battery charge is nonnegative, and below capacity $0\leq q \leq Q\mathbf{1}$.
    * Conservation of charge $q_{t+1}=q_t+c_t$ for $t=1, \ldots T$.
    * Charge at the end = charge at the start: $q_1=q_T+c_T$.

    ## Optimization Variables
    * Battery charging rate $c \in \mathbf{R}^T$
    * Battery charge $q \in \mathbf{R}^T$
    """)
    return


@app.cell
def _(T, cp, np, p, u):
    # Parametrized problem
    q = cp.Variable(shape=(T, 1))
    c = cp.Variable(shape=(T, 1))
    D = cp.Parameter(nonneg=True)
    C = cp.Parameter(nonneg=True)
    Q = cp.Parameter(nonneg=True)
    obj = p.T @ (u + c)
    cons = [c >= -D,
    c <= C,
    q >= 0,
    q <= Q,
    q[1:] == q[:T-1] + c[:T-1],
    q[0] == q[T-1] + c[T-1],
    u + c >= 0]
    ts = np.linspace(1, T, num=T) / 4
    return C, D, Q, c, cons, obj, q, ts


@app.cell
def _(C, D, Q, c, cons, cp, np, obj, p, plt, q, ts):
    def solve_and_plot_duals(Q_val, C_val, D_val):
        Q.value = Q_val
        C.value = C_val
        D.value = D_val
        prob = cp.Problem(cp.Minimize(obj), cons)
        pstar = prob.solve()
        fig = plt.figure(figsize=(8, 7))

        discharge_dual = cons[0].dual_value.flatten()
        charge_dual = cons[1].dual_value.flatten()
        charge_min = cons[2].dual_value.flatten()
        charge_max = cons[3].dual_value.flatten()
        cons_charge = np.zeros_like(charge_min)
        cons_charge[0] = cons[5].dual_value[0]
        cons_charge[1:] = cons[4].dual_value.flatten()
    
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(ts, c.value)
        ax1.plot(ts, discharge_dual * 30)
        ax1.plot(ts, charge_dual* 100)
        ax1.legend(['ct (charging rate)', 'discharge_dual', 'charge_dual'])
    

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(ts, cons_charge)
        ax2.plot(ts, p)
        ax2.legend(['Energy conservation dual', 'price'])

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(ts, q.value)
        ax3.plot(ts, charge_min * 50)
        ax3.plot(ts, charge_max* 100)
        ax3.legend(['Charge upper bound', 'Charge lower bound'])

        fig.tight_layout()
        return fig


    return (solve_and_plot_duals,)


@app.cell
def _(mo):
    mo.md(r"""
    ## What happens as we change the battery parameters?
    """)
    return


@app.cell
def _(mo):
    Q_ui = mo.ui.slider(start=0, stop=150, step=5, value=35, label="Q (capacity)")
    C_ui = mo.ui.slider(start=0, stop=10, step=0.5, value=3, label="C (max charge rate)")
    D_ui = mo.ui.slider(start=0, stop=10, step=0.5, value=3, label="D (max discharge rate)")

    mo.vstack([Q_ui, C_ui, D_ui])
    return C_ui, D_ui, Q_ui


@app.cell
def _(C_ui, D_ui, Q_ui, solve_and_plot_duals):
    Q_val = float(Q_ui.value)
    C_val = float(C_ui.value)
    D_val = float(D_ui.value)

    solve_and_plot_duals(Q_val, C_val, D_val)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations!
    * Dual variables are zero when their corresponding constraints are *slack*
      * For example, when the battery isn't being charged, the corresponding dual value is zero.
      * When the battery is not at full charge, the dual corresponding to the charge upper bound is zero.
    * The energy conservation dual traces out energy prices. Why is that?
    """)
    return


if __name__ == "__main__":
    app.run()
