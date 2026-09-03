"""Locate the horizon where the hybrid stops beating Boris on trajectory.

Writes crossover.json.  Since W6.2 that file also carries `gain_vs_horizon`,
the table this script had only ever printed: the running trajectory advantage
of the projected hybrid over plain Boris at 19.1, 25, 50, 100, 200, 300, 500
and 1000 gyro-orbits.  Section 7 of the manuscript quotes four of those
entries (117.8, 32.7, unity at the crossover, 0.07), and before W6.2 they
lived only in this script's stdout, which is not a data file.
"""
import os, sys, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import fast as F, traj as T

TWO_PI=2*np.pi; DT=F.DT_WORK; TAU=F.TAU_MAIN
H=1e3; n_w=int(round(H*TWO_PI/DT)); refine=150
_,Rr,_,_=F.fine_reference(TAU,DT/refine,n_w*refine,refine)
tg=np.arange(1,n_w+1)*DT/TWO_PI
err={m:np.linalg.norm(T.coarse(m,n_w)-Rr[:n_w],axis=1) for m in ['boris','proj','raw']}
crms={m:np.sqrt(np.cumsum(err[m]**2)/np.arange(1,n_w+1)) for m in err}
ratio=crms['boris']/crms['proj']          # >1 => hybrid better
i=np.where(ratio<1)[0]
res={"crossover_gyrations": float(tg[i[0]]) if len(i) else None,
     "reference_refinement": refine,
     "gain_vs_horizon": []}
print(f"{'горизонт, гир.':>16}{'Boris rms':>13}{'гибрид rms':>13}{'выигрыш':>11}")
for h in [19.1,25,50,100,200,300,500,1000]:
    j=np.searchsorted(tg,h)-1
    if j<1: continue
    res["gain_vs_horizon"].append({
        "gyro_orbits_requested": float(h),
        "gyro_orbits_sampled": float(tg[j]),
        "boris_pos_err_rms": float(crms['boris'][j]),
        "proj_pos_err_rms": float(crms['proj'][j]),
        "raw_pos_err_rms": float(crms['raw'][j]),
        "traj_gain_projected": float(ratio[j])})
    print(f"{h:>16.1f}{crms['boris'][j]:>13.4e}{crms['proj'][j]:>13.4e}{ratio[j]:>11.2f}")
print(f"\nразворот (выигрыш падает до 1): {res['crossover_gyrations']:.1f} гирооборотов" if len(i)
      else "\nразворота нет до 1e3")
for m in ['boris','proj','raw']:
    k=np.where(err[m]>1.0)[0]
    res[f"{m}_reaches_1_larmor_at_gyr"]=float(tg[k[0]]) if len(k) else None
    print(f"{m:6s} достигает 1 ларморовского радиуса при "
          + (f"{tg[k[0]]:.1f} гир." if len(k) else "> 1e3 гир."))
if res["proj_reaches_1_larmor_at_gyr"] and res["boris_reaches_1_larmor_at_gyr"]:
    res["one_larmor_horizon_gain"] = (res["proj_reaches_1_larmor_at_gyr"]
                                      / res["boris_reaches_1_larmor_at_gyr"])
json.dump(res, open(os.path.join(HERE,'crossover.json'),'w'), indent=2)
np.savez_compressed(os.path.join(HERE,'crossover.npz'), t_gyr=tg[::50],
                    **{f'{m}_rms':crms[m][::50] for m in crms})
