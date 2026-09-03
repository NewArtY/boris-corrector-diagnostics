"""Independent verification, T2c: are the two Boris 'variants' the SAME map?

Claim under test (11_THEORY.md, T2c): the shipped variant (v at integer
times, r += v_{n+1} h) and the textbook staggered variant (v at half-integer
times) generate IDENTICAL recursions, differing only in
  (i) the time label attached to the velocity,
  (ii) the initial half-step-back kick,
  (iii) the output synchronization (readout).

Strict test: rename the staggered internal state w_n := v_{n-1/2}.  Then the
staggered update is  w_{n+1} = kick(w_n; E(r_n,t_n), B(r_n,t_n), h),
r_{n+1} = r_n + w_{n+1} h  -- textually the same recursion as the shipped
one.  If the claim is exact, running the staggered core from w_0 = v_0
(i.e. WITHOUT the initial half-back kick) must reproduce the shipped
trajectory to the bit, and running the shipped code from
v_0' = kick(v_0, -h/2) must reproduce the standard staggered internal states
to the bit.  Report max abs differences.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)

from fields import DecayingField
from models.boris import integrate_boris, boris_step

TAU = 1.2e5
H = 0.3
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])


def boris_kick(v, E, B, dt, q=-1.0, m=1.0):
    """Velocity-only Boris kick, written independently (same math as
    experiments/cost/staggered.py:boris_kick)."""
    qmdt2 = 0.5 * q * dt / m
    v_minus = v + qmdt2 * E
    t_vec = qmdt2 * B
    s_vec = 2.0 * t_vec / (1.0 + np.dot(t_vec, t_vec))
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)
    return v_plus + qmdt2 * E


def staggered_core(r0, w0, n_steps, field, h):
    """The staggered-leapfrog recursion with internal state w_n = v_{n-1/2},
    started from an arbitrary internal state w0 (no init convention)."""
    r = np.array(r0, float); w = np.array(w0, float)
    rs = np.zeros((n_steps + 1, 3)); ws = np.zeros((n_steps + 1, 3))
    rs[0], ws[0] = r, w
    t = 0.0
    for i in range(1, n_steps + 1):
        E = np.asarray(field.E(r, t), float)
        B = np.asarray(field.B(r, t), float)
        w = boris_kick(w, E, B, h)
        r = r + w * h
        t += h
        rs[i], ws[i] = r, w
    return rs, ws


def main():
    field = DecayingField(B0=1.0, tau=TAU)
    n = 20000  # ~955 gyroperiods, far beyond the article's 400 steps

    # shipped variant via the repository's own production code
    rs_ship, vs_ship, _ = integrate_boris(R0, V0, 0.0, H, n, field)

    # A) staggered core started from w0 = v0 (init convention removed)
    rs_stag, ws_stag = staggered_core(R0, V0, n, field, H)
    dA_r = float(np.max(np.abs(rs_ship - rs_stag)))
    dA_v = float(np.max(np.abs(vs_ship - ws_stag)))

    # B) shipped code started from the staggered initial state v_{-1/2}
    E0f = np.asarray(field.E(R0, 0.0), float)
    B0f = np.asarray(field.B(R0, 0.0), float)
    v_half_back = boris_kick(V0, E0f, B0f, -0.5 * H)
    rs_ship2, vs_ship2, _ = integrate_boris(R0, v_half_back, 0.0, H, n, field)
    rs_stag2, ws_stag2 = staggered_core(R0, v_half_back, n, field, H)
    dB_r = float(np.max(np.abs(rs_ship2 - rs_stag2)))
    dB_v = float(np.max(np.abs(vs_ship2 - ws_stag2)))

    # C) size of the initial-condition (convention ii) effect on the ORBIT:
    #    standard staggered orbit vs shipped orbit from the same (r0, v0)
    dC_r_rms = float(np.sqrt(np.mean(
        np.sum((rs_ship - rs_ship2) ** 2, axis=1))))

    out = {
        "n_steps": n,
        "A_same_start_no_halfback": {
            "max_abs_dr": dA_r, "max_abs_dv": dA_v,
            "bitwise_identical": bool(dA_r == 0.0 and dA_v == 0.0)},
        "B_same_start_with_halfback": {
            "max_abs_dr": dB_r, "max_abs_dv": dB_v,
            "bitwise_identical": bool(dB_r == 0.0 and dB_v == 0.0)},
        "C_init_convention_orbit_gap_rms_rL": dC_r_rms,
        "note": ("A/B == 0 -> the two recursions are the same map, "
                 "differing only in initial state and readout; "
                 "C is the size of the init-convention effect (O(h))")
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "vt2_map_identity.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
