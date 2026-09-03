"""t2: the energy channel of a leapfrog-type scheme is a property of the
velocity-synchronization convention, not of the map.

Predictions (zero fit parameters):

  P6: staggered Boris with average-recentring  v_n = (v_{n-1/2}+v_{n+1/2})/2
      has relative energy error  sin^2(theta_h/2) = (hOm/2)^2 / (1+(hOm/2)^2)
      per sample, i.e. 2.2005e-2 at h=0.3, Om=1  (article: 2.195e-2 .. 2.20e-2),
      REGARDLESS of trajectory accuracy; scaling h^2 (check h=0.15: 5.594e-3).
  P7: ratio of the two conventions = sin^2(theta_h/2) * 2 tau/h * e^{t/tau}
      ~= Om^2 h tau / 2 = 1.8e4  (article: 1.8e4).
  P8: rotation-recentring (rotate v_{n+1/2} back by theta_h/2) removes the
      artifact entirely -> error collapses to the sampling-offset floor.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 1.2e5
T_FINAL = 120.0


def kick(v, E, B, h):
    k = -0.5 * h
    vm = v + k * E
    tv = k * B
    t2 = tv @ tv
    s = 2 * tv / (1 + t2)
    vp = vm + np.cross(vm, tv)
    return vm + np.cross(vp, s) + k * E


def field(r, t, tau=TAU):
    Bz = np.exp(-t / tau)
    fac = 0.5 * Bz / tau
    return np.array([-fac * r[1], fac * r[0], 0.0]), np.array([0, 0, Bz])


def staggered(h, n, recentre="avg"):
    r = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
    E, B = field(r, 0.0)
    vh = kick(v0, E, B, -0.5 * h)              # v_{-1/2}
    ts = np.zeros(n + 1); vs = np.zeros((n + 1, 3)); vs[0] = v0
    t = 0.0
    for i in range(1, n + 1):
        E, B = field(r, t)
        vh_new = kick(vh, E, B, h)             # v_{n+1/2}
        r = r + vh_new * h
        t += h
        if recentre == "avg":
            vs[i] = 0.5 * (vh + vh_new)
        else:                                   # rotate back by theta_h/2
            Bz = B[2]
            th = -0.5 * 2 * np.arctan(-0.5 * h * Bz)   # half rotation, q=-1
            c, s = np.cos(-th / 1), np.sin(-th)
            # rotate vh_new by -theta_h/2 about z (undo half the step rotation)
            vs[i] = np.array([c * vh_new[0] - s * vh_new[1],
                              s * vh_new[0] + c * vh_new[1], vh_new[2]])
        vh = vh_new
        ts[i] = t
    return ts, vs


out = {}
for h in (0.3, 0.15):
    n = int(round(T_FINAL / h))
    ts, vs = staggered(h, n, "avg")
    E = 0.5 * np.sum(vs**2, axis=1)
    Ephys = 0.5 * np.exp(-ts / TAU)
    err = np.abs(E - Ephys) / 0.5
    half = n // 2
    med = float(np.median(err[half:]))
    pred = float(np.median((np.tan(np.arctan(h * np.exp(-ts / TAU) / 2))**2 /
                            (1 + (h * np.exp(-ts / TAU) / 2)**2) * 0 +
                            np.sin(np.arctan(h * np.exp(-ts / TAU) / 2))**2 *
                            np.exp(-ts / TAU))[half:]))
    # sin^2(theta_h/2) with theta_h = 2 atan(h Om/2); energy scale e^{-t/tau}
    out[f"h={h}"] = {"measured_median": med, "predicted": pred,
                     "article": 2.195e-2 if h == 0.3 else None}

# ratio of conventions at h=0.3
ratio_meas = out["h=0.3"]["measured_median"] / 1.2459e-6
out["convention_ratio"] = {"measured": ratio_meas,
                           "predicted_Om2_h_tau_over_2": 1.0 * 0.3 * TAU / 2,
                           "article": 1.8e4}

# P8: rotation recentring
ts, vs = staggered(0.3, int(round(T_FINAL / 0.3)), "rot")
E = 0.5 * np.sum(vs**2, axis=1)
err = np.abs(E - 0.5 * np.exp(-ts / TAU)) / 0.5
out["rotation_recentring_median"] = float(np.median(err[len(ts) // 2:]))

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "t2_convention.json"), "w"), indent=1)
