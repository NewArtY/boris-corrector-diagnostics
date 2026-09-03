"""Shared harness for the T2 reconciliation.

Two systems, one set of definitions.

SYSTEM M (magnetic).  Boris-type particle in B = B0 f(t) zhat with the
consistent induced E = fac(t) (zhat x r), fac = -0.5 dB/dt.  q = -1, |v0| = 1,
Omega = 1.  Exact continuum energy for the on-axis gyration:
    E_phys(t)/E0 = f(t).
Observable  Q_n = |v_n|^2 / E0 ,  reference  Qref(t) = f(t).

SYSTEM L (Landau-Lifshitz rapidity).
    dtheta/dt = alpha tanh(theta) - eps sinh(theta) cosh(theta)
Substituting y = sinh^2(theta) linearises it to the logistic equation
    dy/dt = Lambda y - 2 eps y^2 ,  Lambda = 2(alpha - eps),
so the system has a CLOSED FORM solution and an exact inverse t(theta).
Observable  Q_n = cosh(theta_n) / cosh(theta_0) = gamma_n / gamma_0,
reference   Qref(t) = gamma_exact(t) / gamma_0.

Both systems therefore expose the same triple (t_n, Q_n, Qref(.)), which is all
the two contested metrics need.

METRIC A -- pure read-out offset (the metric of the second-system agent):
    dev(delta) = median_{n in 2nd half} | Q_n - Qref(t_n + delta) |
    A_signed   = dev(h/2) - dev(0)
    A_ref      = median_{n in 2nd half} | Qref(t_n + h/2) - Qref(t_n) |
A_ref contains NO scheme quantity at all: it is a property of the reference
solution alone.  The identity  dev(h/2) - dev(0) = +/- A_ref  whenever the
scheme error is one-signed is the whole content of "scheme independence".

METRIC B -- floor ratio (the metric of the theory verifier):
    dev0  = dev(0)                        (the scheme's own error)
    R_true = signal(t_med) / dev0         ("600x")
    R_art  = signal(t_med) / A_ref        (the same ratio with the read-out
                                           artefact in the denominator)
    signal(t) = |Qref(t) - Qref(0)|
and both are compared against 2 t_med / h.
"""
import math
import numpy as np

# =============================================================== SYSTEM L ====


class LL:
    """Closed-form Landau-Lifshitz rapidity system."""

    def __init__(self, alpha, eps, theta0):
        self.alpha = alpha
        self.eps = eps
        self.Lam = 2.0 * (alpha - eps)
        self.yinf = (alpha - eps) / eps          # sinh^2(theta*)
        self.theta0 = theta0
        self.y0 = math.sinh(theta0) ** 2
        self.theta_star = math.acosh(math.sqrt(alpha / eps))
        self.g0 = math.cosh(theta0)

    # --- vector field -------------------------------------------------------
    def f(self, th):
        return self.alpha * math.tanh(th) - self.eps * math.sinh(th) * math.cosh(th)

    def fp(self, th):
        ch = math.cosh(th)
        return self.alpha / (ch * ch) - self.eps * math.cosh(2.0 * th)

    # --- exact flow ---------------------------------------------------------
    def y_exact(self, t):
        t = np.asarray(t, float)
        return self.yinf / (1.0 + (self.yinf / self.y0 - 1.0) * np.exp(-self.Lam * t))

    def theta_exact(self, t):
        return np.arcsinh(np.sqrt(self.y_exact(t)))

    def gamma_exact(self, t):
        return np.sqrt(1.0 + self.y_exact(t))

    def Qref(self, t):
        """gamma_exact(t) / gamma_0."""
        return self.gamma_exact(t) / self.g0

    def dQref_dt(self, t):
        y = self.y_exact(t)
        ydot = self.Lam * y - 2.0 * self.eps * y * y
        return ydot / (2.0 * np.sqrt(1.0 + y)) / self.g0

    def t_of_theta(self, th):
        """Exact inverse of the flow: the time at which the exact solution
        started from theta0 reaches theta.  Only valid for y < yinf."""
        y = np.sinh(np.asarray(th, float)) ** 2
        a = self.yinf / self.y0 - 1.0
        return -np.log((self.yinf / y - 1.0) / a) / self.Lam

    # --- one-step schemes ---------------------------------------------------
    def _newton(self, F, dF, x0):
        x = x0
        for _ in range(80):
            d = F(x) / dF(x)
            x -= d
            if abs(d) < 1e-16 * max(1.0, abs(x)):
                break
        return x

    def step(self, name, th, h):
        f, fp = self.f, self.fp
        if name == "euler":
            return th + h * f(th)
        if name == "rk4":
            k1 = f(th); k2 = f(th + .5 * h * k1)
            k3 = f(th + .5 * h * k2); k4 = f(th + h * k3)
            return th + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if name == "ieuler":
            return self._newton(lambda x: x - th - h * f(x),
                                lambda x: 1 - h * fp(x), th + h * f(th))
        if name == "trapezoid":
            c = th + .5 * h * f(th)
            return self._newton(lambda x: x - c - .5 * h * f(x),
                                lambda x: 1 - .5 * h * fp(x), th + h * f(th))
        if name == "midpoint":                      # explicit RK2
            k1 = f(th)
            return th + h * f(th + .5 * h * k1)
        raise ValueError(name)

    def run(self, scheme, h, T, theta0=None):
        th = self.theta0 if theta0 is None else theta0
        n = int(round(T / h)) + 1
        t = np.arange(n) * h
        thn = np.empty(n)
        thn[0] = th
        for i in range(1, n):
            th = self.step(scheme, th, h)
            thn[i] = th
        return t, thn, np.cosh(thn) / self.g0


LL_SCHEMES = ("euler", "midpoint", "trapezoid", "ieuler", "rk4")

# =============================================================== SYSTEM M ====


def mk_field(law, tau, beta=1.0):
    """f(t) = Bz/B0 and fac(t) = -0.5 dBz/dt (so E = fac*(zhat x r))."""
    if law == "exp":
        return (lambda t: np.exp(-np.asarray(t, float) / tau),
                lambda t: 0.5 * np.exp(-np.asarray(t, float) / tau) / tau)
    if law == "pow":
        return (lambda t: (1.0 + np.asarray(t, float) / tau) ** (-beta),
                lambda t: 0.5 * beta / tau * (1.0 + np.asarray(t, float) / tau) ** (-beta - 1.0))
    if law == "lin":
        return (lambda t: 1.0 - np.asarray(t, float) / tau,
                lambda t: 0.5 / tau + 0.0 * np.asarray(t, float))
    if law == "cos":                      # B = 1 - a(1-cos(t/tau)) : Bdot(0)=0
        a = 0.5
        return (lambda t: 1.0 - a * (1.0 - np.cos(np.asarray(t, float) / tau)),
                lambda t: 0.5 * a * np.sin(np.asarray(t, float) / tau) / tau)
    if law == "gauss":                    # Bdot(0) = 0
        return (lambda t: np.exp(-(np.asarray(t, float) / tau) ** 2),
                lambda t: 0.5 * (2 * np.asarray(t, float) / tau ** 2) *
                np.exp(-(np.asarray(t, float) / tau) ** 2))
    raise ValueError(law)


def kick_boris(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    return np.array([vm[0] + vpy * sz + k * E[0],
                     vm[1] - vpx * sz + k * E[1], vm[2]])


def kick_exactrot(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = np.tan(0.5 * q * Bz * h)
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    return np.array([vm[0] + vpy * sz + k * E[0],
                     vm[1] - vpx * sz + k * E[1], vm[2]])


def run_mag(scheme, fB, ffac, h, T, r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), q=-1.0):
    """Return t_n, Q_n = |v_n|^2/E0 (and rho_n) for the requested scheme.

    scheme in {boris_shipped, boris_midpoint_drift, exactrot_shipped, rk4}
    """
    n = int(round(T / h))
    t = np.zeros(n + 1)
    Q = np.zeros(n + 1)
    rho = np.zeros(n + 1)
    if scheme == "rk4":
        y = np.concatenate([np.array(r0, float), np.array(v0, float)])
        E0 = y[3:] @ y[3:]
        Q[0] = 1.0
        rho[0] = math.hypot(y[0], y[1])
        tt = 0.0

        def fode(tt, y):
            r = y[:3]; v = y[3:]
            fac = float(ffac(tt)); Bz = float(fB(tt))
            E = np.array([-fac * r[1], fac * r[0], 0.0])
            return np.concatenate([v, q * (E + np.cross(v, np.array([0.0, 0.0, Bz])))])
        for i in range(1, n + 1):
            k1 = fode(tt, y)
            k2 = fode(tt + h / 2, y + h / 2 * k1)
            k3 = fode(tt + h / 2, y + h / 2 * k2)
            k4 = fode(tt + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            tt += h
            t[i] = tt; Q[i] = (y[3:] @ y[3:]) / E0; rho[i] = math.hypot(y[0], y[1])
        return t, Q, rho

    if scheme.startswith("boris_staggered"):
        # leapfrog variant: internal state w_n := v_{n-1/2}, obtained by a
        # half kick backwards.  Two read-outs of the SAME run:
        #   'avg' : v_n = (w_n + w_{n+1})/2      (average re-centring)
        #   'rot' : |w_{n+1}| = true speed at the half-integer time t_n + h/2
        r = np.array(r0, float); v = np.array(v0, float)
        E0 = v @ v
        E = np.array([-float(ffac(0.0)) * r[1], float(ffac(0.0)) * r[0], 0.0])
        w = kick_boris(v, E, float(fB(0.0)), -0.5 * h, q)
        Q[0] = 1.0
        rho[0] = math.hypot(r[0], r[1])
        tt = 0.0
        for i in range(1, n + 1):
            Bz = float(fB(tt)); fac = float(ffac(tt))
            E = np.array([-fac * r[1], fac * r[0], 0.0])
            wn = kick_boris(w, E, Bz, h, q)
            r = r + wn * h
            tt += h
            if scheme.endswith("avg"):
                vr = 0.5 * (w + wn)
            else:
                vr = wn
            w = wn
            t[i] = tt; Q[i] = (vr @ vr) / E0; rho[i] = math.hypot(r[0], r[1])
        return t, Q, rho

    kick = kick_exactrot if scheme.startswith("exactrot") else kick_boris
    drift = "mid" if "midpoint_drift" in scheme else "new"
    r = np.array(r0, float); v = np.array(v0, float)
    E0 = v @ v
    Q[0] = 1.0
    rho[0] = math.hypot(r[0], r[1])
    tt = 0.0
    for i in range(1, n + 1):
        Bz = float(fB(tt)); fac = float(ffac(tt))
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        vo = v
        v = kick(v, E, Bz, h, q)
        r = r + (v if drift == "new" else 0.5 * (vo + v)) * h
        tt += h
        t[i] = tt; Q[i] = (v @ v) / E0; rho[i] = math.hypot(r[0], r[1])
    return t, Q, rho


MAG_SCHEMES = ("boris_shipped", "boris_midpoint_drift", "exactrot_shipped", "rk4",
               "boris_staggered_avg", "boris_staggered_rot")

# ================================================================ METRICS ====


def metrics(t, Q, Qref, dQref=None, h=None, half=None):
    """Both contested metrics from one (t_n, Q_n, Qref) triple."""
    n = len(t)
    if half is None:
        half = n // 2
    sl = slice(half, None)
    dev0 = float(np.median(np.abs(Q[sl] - Qref(t[sl]))))
    devh = float(np.median(np.abs(Q[sl] - Qref(t[sl] + h / 2.0))))
    devmh = float(np.median(np.abs(Q[sl] - Qref(t[sl] - h / 2.0))))
    A_ref = float(np.median(np.abs(Qref(t[sl] + h / 2.0) - Qref(t[sl]))))
    t_med = float(np.median(t[sl]))
    signal = float(abs(Qref(np.array([t_med]))[0] - Qref(np.array([0.0]))[0]))
    out = {
        "dev0": dev0,
        "dev_plus_h2": devh,
        "dev_minus_h2": devmh,
        "A_signed": devh - dev0,
        "A_ref": A_ref,
        "A_pred_h2_dQdt": (float((h / 2.0) * abs(dQref(np.array([t_med]))[0]))
                           if dQref is not None else None),
        "t_med": t_med,
        "signal": signal,
        "R_true": signal / dev0 if dev0 > 0 else float("inf"),
        "R_art": signal / A_ref if A_ref > 0 else float("inf"),
        "two_t_over_h": 2.0 * t_med / h,
        "R_true_over_2th": (signal / dev0) / (2.0 * t_med / h) if dev0 > 0 else float("inf"),
        "R_art_over_2th": (signal / A_ref) / (2.0 * t_med / h) if A_ref > 0 else float("inf"),
        "floor_in_units_of_artefact": dev0 / A_ref if A_ref > 0 else float("inf"),
    }
    return out


def best_offset(t, Q, Qref, h, half=None, npts=801, span=1.0, zooms=8):
    """Best single GLOBAL read-out offset delta that minimises the reported
    median deviation.  Coarse grid on [-span*h, span*h] followed by repeated
    zooms, so that offsets many orders of magnitude below h are resolved."""
    n = len(t)
    if half is None:
        half = n // 2
    sl = slice(half, None)
    ts = t[sl]
    Qs = Q[sl]

    def obj(d):
        return float(np.median(np.abs(Qs - Qref(ts + d))))

    lo, hi = -span * h, span * h
    best_d, best_v = 0.0, obj(0.0)
    for _ in range(zooms):
        dls = np.linspace(lo, hi, npts)
        vals = np.array([obj(d) for d in dls])
        j = int(np.argmin(vals))
        if vals[j] < best_v:
            best_v, best_d = float(vals[j]), float(dls[j])
        step = dls[1] - dls[0]
        lo, hi = dls[j] - 2 * step, dls[j] + 2 * step
    dev0 = obj(0.0)
    return {"argmin_delta": best_d, "argmin_delta_over_h": best_d / h,
            "min_dev": best_v, "dev0": dev0,
            "collapse_factor": dev0 / max(best_v, 1e-300)}
