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
    # Numerical perturbation analysis

    ![alt](public/image.png)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Putting objective and constraints in matrix form
    1. Rewrite quadratic objective as $x^TQx + f^Tx$ with
    $Q = \begin{bmatrix} 1 \;\; -0.5\\ -0.5 \;\; 2 \end{bmatrix}$ and $f=\begin{bmatrix} -1\\ 0 \end{bmatrix}$.

    2. Rewrite constraints as $Ax-b \leq 0$ with $A = \begin{bmatrix} 1 \;\;\;\;\; 2\\ 1 \;\; -4 \\ 5 \;\;\;\; 76 \end{bmatrix}$ and $b=\begin{bmatrix} u_1\\ u_2 \\ 1 \end{bmatrix}$.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np

    # part (a)
    u1, u2 = -2, -3
    Q = np.array([[1, -0.5], [-0.5, 2]])
    f = np.array([-1, 0])
    A = np.array([[1, 2], [1, -4], [5, 76]])
    b = np.array([u1, u2, 1])
    x = cp.Variable(2)
    obj = cp.quad_form(x, Q) + f @ x 
    cons = [A @ x <= b]

    prob = cp.Problem(cp.Minimize(obj), cons)
    p_star = prob.solve()
    lambdas_unperturbed = cons[0].dual_value
    x_unperturbed = x.value

    print(p_star)
    print(x_unperturbed)
    return (
        A,
        Q,
        b,
        cp,
        f,
        lambdas_unperturbed,
        np,
        obj,
        p_star,
        x,
        x_unperturbed,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Part a: Checking KKT Conditions
    ### 1. Primal Constraints: $f_i(x^*)\leq 0$, and $h_i(x^*)=0$
    We have no equality constraints.

    For the inequality constraints, primal feasability dictates $Ax-b\leq 0$.
    """)
    return


@app.cell
def _(A, b, x_unperturbed):
    print("Ax-b:", A @ x_unperturbed - b)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2. Dual Constraints: $\lambda\geq 0$
    """)
    return


@app.cell
def _(lambdas_unperturbed):
    print(lambdas_unperturbed)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3. Complementary Slackness: $\lambda_if_i(x^*) =0$
    This holds because all the $f_i$'s are zero
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4. Gradient of Lagrangian wrt $x$ vanishes:
    We have $L(x,\lambda) = x^TQx + f^Tx +\lambda^T(Ax-b)$. Taking the gradient, we obtain

    $\lambda^T(Ax) = x^T (A^T\lambda)$


    $\nabla_x L = 2Qx + f+A^T\lambda$
    """)
    return


@app.cell
def _(A, Q, f, lambdas_unperturbed, x_unperturbed):
    print(2 * Q @ x_unperturbed + f + A.T @ lambdas_unperturbed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part b: Perturbed constraints

    $u_1=-2+\delta_1$, $u_2=-3+\delta_2$

    Estimated optimal value: $p^*_{pred} = p^* -\lambda^T\delta$. This is a lower bound for $p^*_{exact}$, obtained from solving the updated optimization problem. **Hint: compare the lagrangians of the perturbed and unperturbed problems.**
    ![alt](public/image_perturb.png)
    """)
    return


@app.cell
def _(A, b, cp, lambdas_unperturbed, np, obj, p_star, x):
    arr_i = np.array([0, -1, 1])
    delta = 0.1
    pa_table = np.zeros((9, 4))
    count = 0
    for i in arr_i:
        for j in arr_i:
            p_pred = p_star - (lambdas_unperturbed[0] * i + lambdas_unperturbed[1] * j) * delta
            cons_perturbed = [A @ x <= b + delta * np.array([i, j, 0])]
            p_exact = cp.Problem(cp.Minimize(obj), cons_perturbed).solve()
            pa_table[count, :] = np.array([i * delta, j * delta, p_pred, p_exact])
            count += 1
    print(pa_table)
    return


if __name__ == "__main__":
    app.run()
