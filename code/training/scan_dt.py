"""Quick diagnostic: how large is the Boris numerical energy error in B4
compared with the physical energy change, as a function of the time step?

This identifies the *degeneracy regime* -- the step size at which numerical
drift becomes comparable to the physical signal. That is the regime in which
the hybrid corrector has to operate.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fields import DecayingField
from models.boris import integrate_boris


def run(dt, t_final, field, r0, v0):
    n = int(round(t_final / dt))
    rs, vs, ts = integrate_boris(r0, v0, 0.0, dt, n, field)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    return ts, E


def main():
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    tau = 150.0
    t_final = 120.0
    field = DecayingField(B0=1.0, tau=tau)

    ts_ref, E_ref = run(0.001, t_final, field, r0, v0)
    E0 = E_ref[0]
    phys = np.abs(E_ref[-1] - E0) / E0
    print(f"physical |dE/E0| over run = {phys:.4e}")

    for dt in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
        ts, E = run(dt, t_final, field, r0, v0)
        Ei = np.interp(ts, ts_ref, E_ref)
        err = np.abs(E - Ei) / E0
        print(f"dt={dt:5.2f} (Om*dt={dt:5.2f})  max|dEnum/E0|={err.max():.3e}  "
              f"final={err[-1]:.3e}  ratio_phys/num={phys/max(err[-1],1e-300):.2e}")


if __name__ == "__main__":
    main()
