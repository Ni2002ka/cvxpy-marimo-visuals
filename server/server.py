from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cvxpy as cp

app = FastAPI()

# TEMP: allow all origins while testing
# Later: set this to your GitHub Pages origin (https://<user>.github.io)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ni2002ka.github.io"],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class SolveRequest(BaseModel):
    positions: list[list[float]]
    max_iter: int | None = 50

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/solve")
def solve(req: SolveRequest):
    P = np.asarray(req.positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        return {"ok": False, "error": "positions must be shape (n,2)"}

    n = P.shape[0]
    if n < 3:
        return {"ok": False, "error": "need at least 3 points"}

    A = cp.Variable((2, 2), PSD=True)
    b = cp.Variable(2)
    cons = [cp.norm2(A @ P[i] + b) <= 1 for i in range(n)]
    prob = cp.Problem(cp.Maximize(cp.log_det(A)), cons)

    try:
        prob.solve(solver=cp.CLARABEL, max_iter=int(req.max_iter or 50))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if A.value is None or b.value is None:
        return {"ok": False, "error": "no solution returned"}

    lambdas = []
    for c in cons:
        dv = c.dual_value
        lambdas.append(float(dv) if dv is not None else 0.0)

    return {"ok": True, "A": A.value.tolist(), "b": b.value.tolist(), "lambdas": lambdas}
