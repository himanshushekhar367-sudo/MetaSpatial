"""Voronoi finite-volume reaction-diffusion solver — numerical verification (physical prior).
Two-point flux approximation on the Delaunay/Voronoi dual mesh; method of manufactured
solutions on [0,1]^2 with u*=sin(pi x)sin(pi y), homogeneous Dirichlet. Reports empirical
convergence order (L2), M-matrix property, and constant-field (discrete-divergence) residual.
Saves fvm_arrays.npz for the figure."""
import numpy as np, json
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
rng=np.random.RandomState(0)
def circum(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    d=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d
    uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d
    return np.array([ux,uy])
def polyarea(P):
    x=P[:,0]; y=P[:,1]; return 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def solve(n,D=1.0,k=0.0,perturb=0.25,want_mat=False):
    g=np.linspace(0,1,n+1); X,Y=np.meshgrid(g,g); pts=np.c_[X.ravel(),Y.ravel()]
    h=1.0/n; bnd=(np.abs(pts[:,0]*(1-pts[:,0]))<1e-12)|(np.abs(pts[:,1]*(1-pts[:,1]))<1e-12)
    pts[~bnd]+=(rng.rand((~bnd).sum(),2)-0.5)*perturb*h          # perturb interior -> genuine Voronoi mesh
    tri=Delaunay(pts); N=len(pts)
    cc=np.array([circum(pts[t[0]],pts[t[1]],pts[t[2]]) for t in tri.simplices])
    from collections import defaultdict
    e2t=defaultdict(list); t2i=defaultdict(list)
    for ti,t in enumerate(tri.simplices):
        for e in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
            e2t[tuple(sorted(e))].append(ti)
        for v in t: t2i[v].append(ti)
    A=lil_matrix((N,N)); area=np.zeros(N)
    for i in range(N):
        if bnd[i]: continue
        P=np.array([cc[t] for t in t2i[i]]); ang=np.arctan2(P[:,1]-pts[i,1],P[:,0]-pts[i,0])
        area[i]=polyarea(P[np.argsort(ang)])
    for (i,j),ts in e2t.items():
        if len(ts)!=2: continue                                  # interior Voronoi face only
        l=np.linalg.norm(cc[ts[0]]-cc[ts[1]]); d=np.linalg.norm(pts[i]-pts[j])
        tr=D*l/d
        for a,b in [(i,j),(j,i)]:
            if not bnd[a]:
                A[a,a]+=tr; A[a,b]-=tr
    for i in range(N):
        if bnd[i]: A[i,i]=1.0
        else: A[i,i]+=k*area[i]
    ue=np.sin(np.pi*pts[:,0])*np.sin(np.pi*pts[:,1])
    f=(2*np.pi**2*D+k)*ue
    rhs=np.zeros(N)
    for i in range(N):
        rhs[i]= ue[i] if bnd[i] else f[i]*area[i]
    # move known Dirichlet neighbours to RHS
    Ac=A.tocsr()
    for i in range(N):
        if bnd[i]: continue
        row=Ac.getrow(i)
        for idx,v in zip(row.indices,row.data):
            if bnd[idx] and idx!=i: rhs[i]-=v*ue[idx]
    Acl=A.copy()                          # independent copy: keep A intact for divergence check
    for i in range(N):
        if bnd[i]: continue
        for idx in list(Acl.rows[i]):
            if bnd[idx]: Acl[i,idx]=0.0
    u=spsolve(Acl.tocsr(),rhs)
    err=np.sqrt(np.sum(area*(u-ue)**2))
    if want_mat:
        Aint=A.tocsr()[np.ix_(~bnd,~bnd)]
        off=Aint-csr_matrix((Aint.diagonal(),(range(Aint.shape[0]),range(Aint.shape[0]))),shape=Aint.shape)
        maxoff=off.max() if off.nnz else 0.0
        dom=np.min(Aint.diagonal()+np.asarray(off.sum(1)).ravel())      # diagonal dominance margin
        # discrete-divergence: full interior rows (incl. boundary columns) on constant field -> k*area
        full=A.tocsr(); rowsum=np.asarray(full@np.ones(N)).ravel()
        const_res=np.abs(rowsum[~bnd]-k*area[~bnd]).max()
        return h,err,u,pts,ue,dict(max_offdiag=float(maxoff),diag_dom_margin=float(dom),const_residual=float(const_res),n_cells=int(N))
    return h,err
hs=[]; errs=[]
for n in [16,24,32,48,64]:
    h,e=solve(n); hs.append(h); errs.append(e); print(f"n={n:3d}  h={h:.4f}  L2err={e:.3e}",flush=True)
hs=np.array(hs); errs=np.array(errs)
order=np.polyfit(np.log(hs),np.log(errs),1)[0]
# worked field + M-matrix stats at moderate resolution
h,e,u,pts,ue,mstat=solve(40,D=1.0,k=2.0,want_mat=True)
orders=np.log(errs[:-1]/errs[1:])/np.log(hs[:-1]/hs[1:])
print(f"\nEmpirical L2 convergence order = {order:.3f}  (pairwise {np.round(orders,2)})")
print(f"M-matrix: max off-diagonal = {mstat['max_offdiag']:.2e} (<=0 required); diagonal-dominance margin = {mstat['diag_dom_margin']:.2e} (>=0)")
print(f"Discrete-divergence (constant-field) residual = {mstat['const_residual']:.2e}")
np.savez_compressed("/home/claude/spatmet/fvm_arrays.npz",hs=hs,errs=errs,order=order,
    pts=pts,u=u,ue=ue,coords=pts)
json.dump(dict(order=float(order),pairwise=list(np.round(orders,3)),**mstat,
    hs=list(np.round(hs,5)),errs=[float(x) for x in errs]),open("/home/claude/spatmet/fvm_summary.json","w"),indent=2)
print("saved fvm_arrays.npz, fvm_summary.json")
