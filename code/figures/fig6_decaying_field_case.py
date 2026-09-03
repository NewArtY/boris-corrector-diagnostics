"""
fig6_decaying_field_case.py
===========================
Figure 4 -- MAIN PHYSICS FIGURE.

What is actually being demonstrated
-----------------------------------
In a slowly decaying magnetic field the decay of B induces an azimuthal
electric field (Faraday's law) that does real work on the particle, so the
energy changes for a genuine physical reason. When the decay is slow this
signal is WEAK, and the question is whether a simulation at a usable time
step can resolve it against its own numerical error.

Measured facts (all numbers produced by this script, none assumed):

  * Plain Boris at Omega_c*dt = 0.3 has an excellent ENERGY budget -- its
    numerical energy error stays ~3 orders below the physical signal -- but
    its TRAJECTORY is badly wrong: the position error accumulates to a
    substantial fraction of the Larmor radius within twenty gyrations, i.e.
    the scheme loses phase coherence. Energy alone therefore certifies
    nothing about where the particle is.

  * The unconstrained learned correction fixes the trajectory but injects
    energy: its numerical energy error rises to the level of the physical
    signal itself, destroying exactly the quantity one wants to measure.

  * Only the CONSTRAINED hybrid -- the learned correction projected
    orthogonal to v with the speed rescaled to the Boris value, so the
    correction cannot change |v| at all -- keeps BOTH error channels below
    the physical signal simultaneously. That joint separation is the result
    of the Article.

Panels
------
(a) The weak physical energy signal (converged fine-step reference).
(b) Numerical energy error against the physical signal level.
(c) Trajectory error against the Larmor radius.

Output: output_figures/Figure4_decaying_field_separation.{png,pdf}
        output_figures/figure4_numbers.json
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import COLORS, FIGURE_DIR, SEED, set_global_seed, get_logger
from figures.plot_style import save_fig, add_panel_label
from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DT_WORK, DT_FINE, T_FINAL, TAU_MAIN
from diagnostics.eval_corrector import load_corrector, integrate_corrected

torch.set_default_dtype(torch.float64)
set_global_seed(SEED)
logger = get_logger("figure4")

C_RAW = "#D55E00"   # unconstrained correction


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    r_larmor = 1.0

    # ---- converged reference: physical truth ----
    n_fine = int(round(T_FINAL / DT_FINE))
    rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1)
    E0 = E_ref[0]
    phys = (E_ref - E0) / E0

    n_work = int(round(T_FINAL / DT_WORK))
    model = load_corrector()

    runs = {
        "boris": integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field),
        "raw": integrate_corrected(field, r0, v0, DT_WORK, n_work, model, project=False),
        "hybrid": integrate_corrected(field, r0, v0, DT_WORK, n_work, model, project=True),
    }

    half = n_work // 2
    res = {}
    for k, (rs, vs, ts) in runs.items():
        E = 0.5 * np.sum(vs ** 2, axis=1)
        Ei = np.interp(ts, ts_r, E_ref)
        e_err = np.abs(E - Ei) / E0
        r_ref_i = np.vstack([np.interp(ts, ts_r, rs_r[:, j]) for j in range(3)]).T
        p_err = np.linalg.norm(rs - r_ref_i, axis=1) / r_larmor
        res[k] = {"ts": ts, "e_err": e_err, "p_err": p_err,
                  "e_med": float(np.median(e_err[half:])),
                  "p_rms": float(np.sqrt(np.mean(p_err ** 2)))}

    ts_b = runs["boris"][2]
    sig = np.abs(np.interp(ts_b, ts_r, phys))
    sig_lvl = float(np.median(sig[half:]))

    for k in res:
        res[k]["S_energy"] = sig_lvl / res[k]["e_med"]
        logger.info("%-7s energy=%.3e (S=%.1f)  traj_rms=%.3e",
                    k, res[k]["e_med"], res[k]["S_energy"], res[k]["p_rms"])
    traj_gain = res["boris"]["p_rms"] / res["hybrid"]["p_rms"]
    logger.info("physical signal=%.3e | trajectory gain = %.0fx (%.2f orders)",
                sig_lvl, traj_gain, np.log10(traj_gain))

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # (a) physical signal
    ax = axes[0]
    ax.plot(ts_r, phys * 100, color=COLORS["physical"], lw=2.0)
    ax.axhline(0, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel("time  $\\Omega_c t$")
    ax.set_ylabel("physical $\\delta E/E_0$  [%]")
    ax.set_title("Weak physical energy signal", fontsize=11)
    ax.text(0.04, 0.08,
            f"slow decay, $\\tau={TAU_MAIN:.1e}$\n"
            f"total change ${phys[-1]*100:.3f}\\,\\%$",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(fc="white", ec=COLORS["physical"], lw=0.8, alpha=0.95))

    # (b) energy error channel
    ax = axes[1]
    ax.axhline(sig_lvl, color=COLORS["physical"], lw=2.0, ls="--")
    ax.axhspan(sig_lvl, sig_lvl * 1e3, color=COLORS["physical"], alpha=0.10)
    for k, c, lab in [("boris", COLORS["boris"], "Boris"),
                      ("raw", C_RAW, "unconstrained correction"),
                      ("hybrid", COLORS["boris_corrector"], "constrained hybrid")]:
        ax.semilogy(res[k]["ts"], np.maximum(res[k]["e_err"], 1e-18), color=c,
                    lw=1.3, label=lab)
    ax.set_xlabel("time  $\\Omega_c t$")
    ax.set_ylabel("numerical $|\\delta E_{\\rm num}/E_0|$")
    ax.set_title("Energy channel", fontsize=11)
    ax.set_ylim(1e-8, sig_lvl * 3e2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.text(0.03, 0.94, "physical signal level", transform=ax.transAxes,
            fontsize=8, va="top", color=COLORS["physical"])

    # (c) trajectory error channel
    ax = axes[2]
    ax.axhline(1.0, color=COLORS["physical"], lw=2.0, ls="--")
    ax.axhspan(1.0, 1e3, color=COLORS["physical"], alpha=0.10)
    for k, c, lab in [("boris", COLORS["boris"], "Boris"),
                      ("raw", C_RAW, "unconstrained correction"),
                      ("hybrid", COLORS["boris_corrector"], "constrained hybrid")]:
        ax.semilogy(res[k]["ts"], np.maximum(res[k]["p_err"], 1e-18), color=c,
                    lw=1.3, label=lab)
    ax.set_xlabel("time  $\\Omega_c t$")
    ax.set_ylabel("position error  $|\\Delta r|/r_L$")
    ax.set_title("Trajectory channel", fontsize=11)
    ax.set_ylim(1e-6, 5.0)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.text(0.03, 0.94, "one Larmor radius", transform=ax.transAxes,
            fontsize=8, va="top", color=COLORS["physical"])

    for a, l in zip(axes, "abc"):
        add_panel_label(a, l)
    fig.tight_layout()
    save_fig(fig, "Figure4_decaying_field_separation")

    numbers = {
        "regime": {"omega_c_dt": DT_WORK, "tau": TAU_MAIN, "t_final": T_FINAL,
                   "n_gyrations": round(T_FINAL / (2 * np.pi), 1)},
        "physical_signal_median": sig_lvl,
        "energy_error_median": {k: res[k]["e_med"] for k in res},
        "energy_separation_ratio": {k: res[k]["S_energy"] for k in res},
        "trajectory_rms_error_in_larmor_radii": {k: res[k]["p_rms"] for k in res},
        "trajectory_gain_hybrid_over_boris": float(traj_gain),
        "trajectory_gain_orders": float(np.log10(traj_gain)),
    }
    with open(os.path.join(FIGURE_DIR, "figure4_numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
