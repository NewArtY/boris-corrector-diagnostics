"""Trajectory error vs fine reference at 1e3 / 1e4 gyrations."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import fast as F

TWO_PI = 2*np.pi; DT = F.DT_WORK; TAU = F.TAU_MAIN
Ws,bs,xm,xs,ysc,_ = F.load_net_numpy(); fwd = F.make_forward(Ws,bs,xm,xs,ysc)


def coarse(mode, n_steps):
    rx,ry,rz=1.0,0.0,0.0; vx,vy,vz=0.0,1.0,0.0; t=0.0; k=-0.5*DT; it=1.0/TAU
    R=np.zeros((n_steps,3)); x=np.empty(13)
    for i in range(n_steps):
        Bz=np.exp(-t*it); fac=0.5*Bz*it; Ex=-fac*ry; Ey=fac*rx
        if mode!='boris':
            x[:] = (rx,ry,rz,vx,vy,vz,0.0,0.0,Bz,Ex,Ey,0.0,DT); d=fwd(x)
        kEx=k*Ex; kEy=k*Ey; vmx=vx+kEx; vmy=vy+kEy; vmz=vz
        tz=k*Bz; sz=2.0*tz/(1.0+tz*tz)
        vpx=vmx+vmy*tz; vpy=vmy-vmx*tz
        vbx=vmx+vpy*sz+kEx; vby=vmy-vpx*sz+kEy; vbz=vmz
        rbx=rx+vbx*DT; rby=ry+vby*DT; rbz=rz+vbz*DT
        if mode=='boris': rx,ry,rz,vx,vy,vz=rbx,rby,rbz,vbx,vby,vbz
        else:
            dvx,dvy,dvz=d[3],d[4],d[5]
            if mode=='proj':
                nb=np.sqrt(vbx*vbx+vby*vby+vbz*vbz); inb=1.0/max(nb,1e-300)
                hx,hy,hz=vbx*inb,vby*inb,vbz*inb; dot=dvx*hx+dvy*hy+dvz*hz
                dvx-=dot*hx; dvy-=dot*hy; dvz-=dot*hz
                nvx,nvy,nvz=vbx+dvx,vby+dvy,vbz+dvz
                sc=nb/max(np.sqrt(nvx*nvx+nvy*nvy+nvz*nvz),1e-300)
                vx,vy,vz=nvx*sc,nvy*sc,nvz*sc
            else: vx,vy,vz=vbx+dvx,vby+dvy,vbz+dvz
            rx,ry,rz=rbx+d[0],rby+d[1],rbz+d[2]
        t+=DT; R[i]=(rx,ry,rz)
    return R

# The driver is guarded so that `import traj` (crossover.py needs only
# `coarse`) does not silently recompute traj_summary.json, which costs four
# minutes and was the reason a stale summary could survive a rerun.
if __name__ == "__main__":
    out={}
    for H, refine in [(1e3,150),(1e3,1500),(1e4,150)]:
        n_w=int(round(H*TWO_PI/DT)); dtf=DT/refine
        t0=time.time(); _,Rr,_,_=F.fine_reference(TAU,dtf,n_w*refine,refine); tref=time.time()-t0
        key=f"H{H:.0e}_ref{refine}x"; out[key]={"n_coarse":n_w,"refine":refine,"dt_fine":dtf,"ref_seconds":tref}
        for mode in ['boris','raw','proj']:
            R=coarse(mode,n_w); e=np.linalg.norm(R-Rr[:n_w],axis=1)
            out[key][mode]={"pos_err_rms":float(np.sqrt(np.mean(e**2))),"pos_err_final":float(e[-1])}
        g=out[key]['boris']['pos_err_rms']/out[key]['proj']['pos_err_rms']
        out[key]['traj_gain_projected']=float(g)
        print(f"{H:.0e} гир., эталон {refine}x ({tref:.0f}s):  "
              f"boris={out[key]['boris']['pos_err_rms']:.4e}  "
              f"proj={out[key]['proj']['pos_err_rms']:.4e}  "
              f"raw={out[key]['raw']['pos_err_rms']:.4e}  выигрыш={g:.1f}x")
    json.dump(out, open(os.path.join(HERE,'traj_summary.json'),'w'), indent=2)
    a=out['H1e+03_ref150x']['boris']['pos_err_rms']; b=out['H1e+03_ref1500x']['boris']['pos_err_rms']
    print(f"\nсходимость эталона (Boris rms при 1e3, 150x против 1500x): {a:.6e} / {b:.6e}, отн.разн.={abs(a-b)/b:.2e}")
