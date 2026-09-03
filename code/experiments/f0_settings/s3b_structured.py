"""
s3b_structured.py -- S3 given its best shot, and measured properly.
====================================================================
The first S3 run used a spatially UNIFORM dB_z. That kills the setting by
construction: a ponderomotive (mean) force needs a gradient of the oscillating
field, so a uniform zero-mean dB can only produce diffusion, never a mean
shift. Ignoring dB entirely then costs 2% of the signal, and there is nothing
for an effective model to learn.

Two fixes here, both needed for an honest verdict.

1. SPATIAL STRUCTURE. dA_phi(rho,t) = a(t) h(rho), h(rho) = (rho/2) exp(-rho^2/2w^2),
   so dB_z = a(t) exp(-rho^2/2w^2) (1 - rho^2/2w^2) and dE_phi = -a'(t) h(rho).
   Still axisymmetric (p_phi stays meaningful, as the plan requires) and still
   Maxwell-consistent, but now the particle samples a gradient.

2. ANTITHETIC VARIATES. The mean effect of dB is second order in amplitude and
   was buried under the realization spread: with sd ~ 4.6e-3 and 24 samples the
   standard error was 9.4e-4, larger than the effect being measured, so the
   first scan could not resolve it at all. Running each realization together
   with its sign-flipped twin (phases -> phases + pi, i.e. a -> -a) cancels the
   O(a) term exactly in the pair mean and leaves the O(a^2) term with a tiny
   variance. This is what makes the question answerable within the budget.

The verdict question is unchanged: is the ensemble-averaged dynamics anything
other than the smooth dynamics plus a two-parameter law?
"""
import json
import os
import numpy as np

import harness as H
from pfields import PerturbedDecaying, Broadband

HERE = os.path.dirname(os.path.abspath(__file__))
DT_FINE = 0.05
W_GRAD = 1.0            # gradient scale, comparable to the Larmor radius


class StructuredPerturbed:
    """B_smooth + spatially structured dB, both from a vector potential."""

    def __init__(self, B0=1.0, tau=H.TAU_MAIN, a=None, adot=None, w=W_GRAD,
                 sign=1.0):
        self.B0, self.tau, self.w, self.sign = float(B0), float(tau), float(w), float(sign)
        self._a, self._adot = a, adot

    def _rho2(self, r):
        return r[:, 0] ** 2 + r[:, 1] ** 2

    def B(self, r, t):
        r = np.atleast_2d(r).astype(float)
        out = np.zeros((r.shape[0], 3))
        bz = self.B0 * np.exp(-float(t) / self.tau)
        if self._a is not None:
            s = self._rho2(r) / (2.0 * self.w ** 2)
            bz = bz + self.sign * float(self._a(t)) * np.exp(-s) * (1.0 - s)
        out[:, 2] = bz
        return out if r.shape[0] > 1 else out[0]

    def E(self, r, t):
        r = np.atleast_2d(r).astype(float)
        # smooth part: E_phi/rho = -0.5 dBz/dt
        factor = -0.5 * (-(self.B0 / self.tau) * np.exp(-float(t) / self.tau))
        out = np.zeros((r.shape[0], 3))
        out[:, 0] = -factor * r[:, 1]
        out[:, 1] = factor * r[:, 0]
        if self._adot is not None:
            # dE_phi = -a'(t) h(rho),  h = (rho/2) exp(-rho^2/2w^2)
            s = self._rho2(r) / (2.0 * self.w ** 2)
            coef = -self.sign * float(self._adot(t)) * 0.5 * np.exp(-s)
            out[:, 0] += -coef * r[:, 1]
            out[:, 1] += coef * r[:, 0]
        return out if r.shape[0] > 1 else out[0]


def dE_final(field, dt=DT_FINE):
    n = int(round(H.T_FINAL / dt))
    rs, vs, ts, fl = H.run_scheme("vps4", field, dt, n)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    return float((E[-1] - E[0]) / E[0]), fl


def antithetic_mean(rms, n_pairs, structured=True, seed0=0):
    """Mean dE over sign-flipped pairs; the O(a) term cancels in each pair."""
    vals = []
    for k in range(n_pairs):
        p = Broadband(w_lo=0.5, w_hi=5.0, n_modes=64, rms=rms, seed=seed0 + k)
        pair = []
        for sgn in (+1.0, -1.0):
            if structured:
                f = StructuredPerturbed(a=p.a, adot=p.adot, sign=sgn)
            else:
                f = PerturbedDecaying(a=(lambda t, s=sgn, q=p: s * q.a(t)),
                                      adot=(lambda t, s=sgn, q=p: s * q.adot(t)))
            v, _ = dE_final(f)
            pair.append(v)
        vals.append(0.5 * (pair[0] + pair[1]))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))), vals


def main():
    out = {"meta": {"setting": "S3b", "w_grad": W_GRAD, "dt": DT_FINE,
                    "method": "antithetic pairs (a, -a) to isolate O(a^2)"}}

    # baseline: the smooth field alone
    smooth = StructuredPerturbed()
    dE_smooth, fl_one = dE_final(smooth)
    out["smooth_only"] = {"dE_final": dE_smooth, "flops": fl_one}
    print(f"smooth-only dE(T) = {dE_smooth:+.6e}   ({fl_one:.3e} flops/run)")

    n_pairs = 24
    rows = []
    for rms in (3e-4, 1e-3, 3e-3, 1e-2):
        for structured in (False, True):
            m, se, vals = antithetic_mean(rms, n_pairs, structured=structured)
            shift = m - dE_smooth
            rows.append({"rms": rms, "structured": structured,
                         "mean_dE": m, "stderr": se,
                         "mean_shift_vs_smooth": shift,
                         "shift_over_stderr": shift / se if se else float("nan"),
                         "shift_over_signal": shift / abs(dE_smooth)})
            tag = "structured" if structured else "uniform   "
            print(f"  rms={rms:.1e} {tag}  shift={shift:+.3e} "
                  f"+-{se:.1e}  ({shift / se if se else 0:+.1f} sigma, "
                  f"{shift / abs(dE_smooth):+.4f} of signal)")
    out["mean_shift"] = rows

    with open(os.path.join(HERE, "s3b_structured.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote s3b_structured.json")


if __name__ == "__main__":
    main()
