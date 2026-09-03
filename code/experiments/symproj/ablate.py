"""Which component of the correction drives the secular energy drift?"""
import os, sys, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import symproj as S
TWO_PI=2*np.pi; DT=S.DT_WORK; TAU=S.TAU_MAIN
N=int(round(100000*TWO_PI/DT))
fwd=S.load_forward()
# Labels are the JSON keys and are English on purpose: this file is read by a
# referee, and until W6.2 it was written with Russian keys AND, because the
# stream was opened without an encoding, in the platform code page -- so
# json.load(open(f, encoding="utf-8")) raised UnicodeDecodeError on it, and
# every sweep that walks the bundle's JSON skipped it silently.
cases=[("proj",False,False,"full correction"),
       ("proj",True,False,"position correction removed"),
       ("proj",False,True,"velocity correction removed"),
       ("sym", True,False,"symmetric projection, position correction removed")]
out={"config":{"dt":DT,"tau":TAU,"n_gyrations":100000,"n_steps":N,
               "base":"shipped"}}
print(f"{'вариант':32s}{'показатель':>12s}{'E_err(1e5)':>14s}")
for mode,zr,zv,label in cases:
    d=S.run(mode,TAU,DT,N,fwd=fwd,base="shipped",n_samples=4000,zero_dr=zr,zero_dv=zv)
    env=np.maximum.accumulate(d["env"]); e=S.envelope_exponent(d["t"],env)
    out[label]={"mode":mode,"zero_dr":zr,"zero_dv":zv,
                "envelope_exponent":float(e),"energy_err_max_1e5":float(env[-1])}
    print(f"{label:32s}{e:>12.3f}{env[-1]:>14.4e}")
json.dump(out,open(os.path.join(HERE,"ablation.json"),"w",encoding="utf-8"),
          indent=2,ensure_ascii=False)
