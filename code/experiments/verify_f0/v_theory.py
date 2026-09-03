"""Analytic verification (sympy) of:
  (7)  DEL linearity in Delta for L_d = |D|^2/(2h) - A(qm,tm).D, symmetric gauge
       -> p_k = M D + b, p_{k+1} = D/h + a(qk_y,-qk_x,0), D1D2 L_d = -M
  (10) singular values of M and cond = sqrt(1+(Omega_c h/2)^2)
"""
import sympy as sp

h, a = sp.symbols('h a', positive=True)
qkx, qky, qkz, q1x, q1y, q1z = sp.symbols('qkx qky qkz q1x q1y q1z')
qk = sp.Matrix([qkx, qky, qkz]); q1 = sp.Matrix([q1x, q1y, q1z])
D = q1 - qk
qm = (qk + q1) / 2

# A(q) = a*(-q_y, q_x, 0), a = Bz(tm)/2 (charge q_c=-1 folded in per varint.py:
# L = |qdot|^2/2 - A.qdot)
A = sp.Matrix([-a*qm[1], a*qm[0], 0])
Ld = (D.dot(D))/(2*h) - A.dot(D)

D1 = sp.Matrix([sp.diff(Ld, v) for v in (qkx, qky, qkz)])   # dLd/dq_k
D2 = sp.Matrix([sp.diff(Ld, v) for v in (q1x, q1y, q1z)])   # dLd/dq_{k+1}

M = sp.Matrix([[1/h, a, 0], [-a, 1/h, 0], [0, 0, 1/h]])
b = sp.Matrix([a*qky, -a*qkx, 0])

print("check p_k = -D1 Ld == M*D + b :", sp.simplify(-D1 - (M*D + b)))
print("check p_k1 = D2 Ld == D/h + a*(qk_y,-qk_x,0) :",
      sp.simplify(D2 - (D/h + sp.Matrix([a*qky, -a*qkx, 0]))))

# linearity: -D1 Ld must be linear in D (jacobian wrt q1 constant)
J = sp.simplify(sp.Matrix([[sp.diff(-D1[i], v) for v in (q1x, q1y, q1z)]
                           for i in range(3)]))
print("d(-D1 Ld)/dq1 (should be M, constant):"); sp.pprint(J)
print("equals M:", sp.simplify(J - M))

# D1D2 Ld: mixed second derivative
D1D2 = sp.simplify(sp.Matrix([[sp.diff(D1[i], v) for v in (q1x, q1y, q1z)]
                              for i in range(3)]))
print("D1D2 Ld (should be -M):", sp.simplify(D1D2 + M))

# singular values of M
MTM = sp.simplify(M.T * M)
print("M^T M:"); sp.pprint(MTM)
ev = MTM.eigenvals()
print("eigenvalues of M^T M:", ev)
cond = sp.sqrt(sp.Rational(1) + a**2*h**2)
print("cond formula sqrt(1+(a h)^2) with a=Omega_c/2 -> sqrt(1+(Om h/2)^2)")
import math
for Oh in (0.3, 1.0, math.pi, 2*math.pi, 4*math.pi, 30.0):
    print(f"  Om_c h={Oh:8.4f}  cond={math.sqrt(1+(Oh/2)**2):.4f}")
