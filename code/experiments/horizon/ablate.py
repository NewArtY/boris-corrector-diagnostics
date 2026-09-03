"""Mechanism ablation (dr=0) + second corrector model on long horizons."""
import os, sys, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import fast as F
from common import CHECKPOINT_DIR

TWO_PI = 2*np.pi; DT = F.DT_WORK
N_GYR = 100000; N_STEPS = int(round(N_GYR*TWO_PI/DT))
HOR = [1e3, 1e4, 1e5]


def run(mode, tau, dt, n_steps, fwd, zero_dr=False, n_samples=4000):
    rx, ry, rz = 1.0, 0.0, 0.0; vx, vy, vz = 0.0, 1.0, 0.0
    t = 0.0; k = -0.5*dt; it = 1.0/tau; E0 = 0.5
    stride = max(1, n_steps//n_samples)
    ts, es, mus, envs = [], [], [], []; rmax = 0.0
    x = np.empty(13 if fwd[1] == 13 else 10); fn = fwd[0]
    for i in range(1, n_steps+1):
        Bz = np.exp(-t*it); fac = 0.5*Bz*it
        Ex = -fac*ry; Ey = fac*rx
        if fwd[1] == 13:
            x[:] = (rx,ry,rz,vx,vy,vz,0.0,0.0,Bz,Ex,Ey,0.0,dt)
        else:
            x[:] = (rx,ry,rz,vx,vy,vz,0.0,0.0,Bz,dt)
        d = fn(x)
        kEx = k*Ex; kEy = k*Ey
        vmx = vx+kEx; vmy = vy+kEy; vmz = vz
        tz = k*Bz; sz = 2.0*tz/(1.0+tz*tz)
        vpx = vmx+vmy*tz; vpy = vmy-vmx*tz
        vbx = vmx+vpy*sz+kEx; vby = vmy-vpx*sz+kEy; vbz = vmz
        rbx = rx+vbx*dt; rby = ry+vby*dt; rbz = rz+vbz*dt
        dvx, dvy, dvz = d[3], d[4], d[5]
        if mode == 'proj':
            nb = np.sqrt(vbx*vbx+vby*vby+vbz*vbz); inb = 1.0/max(nb,1e-300)
            hx, hy, hz = vbx*inb, vby*inb, vbz*inb
            dot = dvx*hx+dvy*hy+dvz*hz
            dvx -= dot*hx; dvy -= dot*hy; dvz -= dot*hz
            nvx, nvy, nvz = vbx+dvx, vby+dvy, vbz+dvz
            sc = nb/max(np.sqrt(nvx*nvx+nvy*nvy+nvz*nvz),1e-300)
            vx, vy, vz = nvx*sc, nvy*sc, nvz*sc
        else:
            vx, vy, vz = vbx+dvx, vby+dvy, vbz+dvz
        if zero_dr: rx, ry, rz = rbx, rby, rbz
        else: rx, ry, rz = rbx+d[0], rby+d[1], rbz+d[2]
        t += dt
        Ec = 0.5*(vx*vx+vy*vy+vz*vz); Bc = np.exp(-t*it); Ep = E0*np.exp(-t*it)
        dev = abs(Ec-Ep)/E0
        if dev > rmax: rmax = dev
        if i % stride == 0 or i == n_steps:
            ts.append(t); es.append(dev); mus.append(abs((Ec/Bc)/(E0)-1.0)); envs.append(rmax); rmax = 0.0
    return np.array(ts), np.array(es), np.array(mus), np.maximum.accumulate(np.array(envs))


Ws,bs,xm,xs,ysc,_ = F.load_net_numpy()
fwd_b4 = (F.make_forward(Ws,bs,xm,xs,ysc), 13)

sd = torch.load(os.path.join(CHECKPOINT_DIR,'boris_corrector.pt'), map_location='cpu')
W2 = [sd[f'net.{i}.weight'].numpy().astype(np.float64) for i in (0,2,4,6)]
b2 = [sd[f'net.{i}.bias'].numpy().astype(np.float64) for i in (0,2,4,6)]
def fwd2(x):
    z = x
    for i in range(3): z = np.tanh(W2[i] @ z + b2[i])
    return 0.05*(W2[3] @ z + b2[3])       # CORRECTION_SCALE = 0.05
fwd_gen = (fwd2, 10)

CASES = [
 ("proj_dr0__paper",      'proj', F.TAU_MAIN, fwd_b4, True),
 ("proj_dr0__quasistatic",'proj', 1.2e8,      fwd_b4, True),
 ("genmodel_raw__paper",  'raw',  F.TAU_MAIN, fwd_gen, False),
 ("genmodel_proj__paper", 'proj', F.TAU_MAIN, fwd_gen, False),
]
out, store = {}, {}
for name, mode, tau, fwd, zdr in CASES:
    t0 = time.time(); t,e,mu,env = run(mode,tau,DT,N_STEPS,fwd,zero_dr=zdr); el = time.time()-t0
    store[name+'/t']=t; store[name+'/e_err']=e; store[name+'/env']=env; store[name+'/mu_err']=mu
    sel = (t > t[-1]/100) & (env > 0)
    expo = float(np.polyfit(np.log10(t[sel]), np.log10(env[sel]),1)[0]) if sel.sum()>10 else float('nan')
    phys_t = 1-np.exp(-t/tau); cr = np.where(e>phys_t)[0]
    rec = {"exponent": expo, "seconds": el,
           "cross_gyrations": float(t[cr[0]]/TWO_PI) if len(cr) else None, "horizons": {}}
    for H in HOR:
        m = t <= H*TWO_PI
        if m.sum()<5: continue
        rec["horizons"][f"{H:.0e}"] = {"energy_err_max": float(env[m][-1]),
            "mu_err_max": float(np.max(mu[m])), "physical_signal": float(1-np.exp(-H*TWO_PI/tau))}
    out[name] = rec
    c = rec["cross_gyrations"]
    print(f"{name:24s} exp={expo:6.3f}  E_err(1e5)={rec['horizons']['1e+05']['energy_err_max']:.3e}  "
          f"пересечение={'нет' if c is None else f'{c:.4g} гир.'}  ({el:.0f}s)")

np.savez_compressed(os.path.join(HERE,'ablate.npz'), **store)
json.dump(out, open(os.path.join(HERE,'ablate_summary.json'),'w'), indent=2)
print('\nсохранено: ablate.npz, ablate_summary.json')
