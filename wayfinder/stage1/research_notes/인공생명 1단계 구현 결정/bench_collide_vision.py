import numpy as np, time, math
from scipy.spatial import cKDTree
import numba as nb

def timeit(f, n=7):
    f(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return 1e3*np.median(ts)

# ---------- numpy padded cell table ----------
def build_table(pos, cs, L):
    G=int(math.ceil(L/cs)); ci=np.minimum((pos/cs).astype(np.int64),G-1)
    cid=ci[:,0]*G+ci[:,1]
    order=np.argsort(cid,kind='stable'); sc=cid[order]
    start=np.searchsorted(sc,sc,side='left'); rank=np.arange(len(sc))-start
    K=int(rank.max())+1
    table=np.full((G*G+1,K),-1,dtype=np.int64)   # last row = dummy empty cell
    table[sc,rank]=order
    return table,ci,G,K

OFF=np.array([(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)])
def cand_from_table(table,ci,G):
    nx=ci[:,None,0]+OFF[None,:,0]; ny=ci[:,None,1]+OFF[None,:,1]
    ok=(nx>=0)&(nx<G)&(ny>=0)&(ny<G)
    ncid=np.where(ok,nx*G+ny,G*G)
    return table[ncid].reshape(len(ci),-1)       # (N, 9K)

def np_collide(pos,R,L):
    table,ci,G,K=build_table(pos,2*R,L)
    cand=cand_from_table(table,ci,G)
    d=pos[cand]-pos[:,None,:]                    # cand=-1 -> garbage, masked
    dist2=(d*d).sum(-1)
    idx=np.arange(len(pos))[:,None]
    hit=(cand>idx)&(dist2<(2*R)**2)
    i,k=np.nonzero(hit)
    return i,cand[i,k]

# ---------- numba cell list ----------
@nb.njit(cache=False)
def nb_build(pos,cs,G):
    N=pos.shape[0]; cid=np.empty(N,np.int64); cnt=np.zeros(G*G+1,np.int64)
    for i in range(N):
        cx=min(int(pos[i,0]/cs),G-1); cy=min(int(pos[i,1]/cs),G-1)
        cid[i]=cx*G+cy; cnt[cid[i]+1]+=1
    start=np.cumsum(cnt); fill=start[:-1].copy(); order=np.empty(N,np.int64)
    for i in range(N):
        order[fill[cid[i]]]=i; fill[cid[i]]+=1
    return cid,start,order

@nb.njit(cache=False)
def nb_collide(pos,R,L,out):
    cs=2*R; G=int(math.ceil(L/cs)); cid,start,order=nb_build(pos,cs,G)
    n=0; r2=(2*R)**2
    for i in range(pos.shape[0]):
        cx=cid[i]//G; cy=cid[i]%G
        for dx in range(-1,2):
            x=cx+dx
            if x<0 or x>=G: continue
            for dy in range(-1,2):
                y=cy+dy
                if y<0 or y>=G: continue
                c=x*G+y
                for s in range(start[c],start[c+1]):
                    j=order[s]
                    if j>i:
                        ddx=pos[j,0]-pos[i,0]; ddy=pos[j,1]-pos[i,1]
                        if ddx*ddx+ddy*ddy<r2:
                            if n<out.shape[0]:
                                out[n,0]=i; out[n,1]=j
                            n+=1
    return n

# ---------- vision: N agents x Rr rays vs circles ----------
def np_vision(pos,head,R,L,rng_,nray,fov):
    table,ci,G,K=build_table(pos,rng_,L)
    cand=cand_from_table(table,ci,G)             # (N,C)
    N=len(pos); idx=np.arange(N)[:,None]
    valid=(cand>=0)&(cand!=idx)
    m=pos[cand]-pos[:,None,:]                    # (N,C,2)
    ang=head[:,None]+np.linspace(-fov/2,fov/2,nray)[None,:]
    dx=np.cos(ang); dy=np.sin(ang)               # (N,Rr)
    b=m[:,None,:,0]*dx[:,:,None]+m[:,None,:,1]*dy[:,:,None]   # (N,Rr,C)
    c2=(m*m).sum(-1)-R*R                         # (N,C)
    disc=b*b-c2[:,None,:]
    t=b-np.sqrt(np.maximum(disc,0))
    ok=valid[:,None,:]&(disc>=0)&(t>0)&(t<=rng_)
    t=np.where(ok,t,np.inf)
    k=t.argmin(-1); tmin=np.take_along_axis(t,k[...,None],-1)[...,0]
    hit=np.take_along_axis(np.broadcast_to(cand[:,None,:],t.shape),k[...,None],-1)[...,0]
    return tmin,np.where(np.isfinite(tmin),hit,-1)

@nb.njit(parallel=True,cache=False)
def nb_vision(pos,head,R,L,rng_,nray,fov,tout,hout):
    cs=rng_; G=int(math.ceil(L/cs)); cid,start,order=nb_build(pos,cs,G)
    N=pos.shape[0]
    for i in nb.prange(N):
        cx=cid[i]//G; cy=cid[i]%G
        for r in range(nray):
            a=head[i]+(-fov/2+fov*r/(nray-1)); dx=math.cos(a); dy=math.sin(a)
            best=np.inf; bj=-1
            for ox in range(-1,2):
                x=cx+ox
                if x<0 or x>=G: continue
                for oy in range(-1,2):
                    y=cy+oy
                    if y<0 or y>=G: continue
                    c=x*G+y
                    for s in range(start[c],start[c+1]):
                        j=order[s]
                        if j==i: continue
                        mx=pos[j,0]-pos[i,0]; my=pos[j,1]-pos[i,1]
                        b=mx*dx+my*dy
                        if b<=0: continue
                        disc=b*b-(mx*mx+my*my-R*R)
                        if disc<0: continue
                        t=b-math.sqrt(disc)
                        if t>0 and t<=rng_ and t<best:
                            best=t; bj=j
            tout[i,r]=best; hout[i,r]=bj

nb_vision_st=nb.njit(parallel=False)(nb_vision.py_func)

print("numba threads:",nb.get_num_threads())
R=1.0; phi=0.05; VR=15.0; fov=math.radians(120)
rs=np.random.default_rng(0)
for N in (1000,3000,10000):
    L=math.sqrt(N*math.pi*R*R/phi)
    pos=rs.random((N,2))*L; head=rs.random(N)*2*math.pi
    out=np.empty((8*N,2),np.int64)
    def kd():
        t=cKDTree(pos); return t.query_pairs(2*R,output_type='ndarray')
    pk=kd(); i1,j1=np_collide(pos,R,L); n2=nb_collide(pos,R,L,out)
    assert len(pk)==len(i1)==n2,(len(pk),len(i1),n2)
    print(f"N={N} L={L:.0f} pairs={n2} | cKDTree build+query_pairs {timeit(kd):.2f} ms | numpy cell-table {timeit(lambda:np_collide(pos,R,L)):.2f} ms | numba cell-list {timeit(lambda:nb_collide(pos,R,L,out)):.3f} ms")
    for nray in (8,16,32):
        tout=np.empty((N,nray)); hout=np.empty((N,nray),np.int64)
        t_np,h_np=np_vision(pos,head,R,L,VR,nray,fov)
        nb_vision(pos,head,R,L,VR,nray,fov,tout,hout)
        assert np.allclose(np.where(np.isfinite(t_np),t_np,-1),np.where(np.isfinite(tout),tout,-1)), "vision mismatch"
        tout2=tout.copy(); hout2=hout.copy()
        print(f"   vision rays={nray} range={VR} hitfrac={np.isfinite(tout).mean():.2f} | numpy {timeit(lambda:np_vision(pos,head,R,L,VR,nray,fov),3):.1f} ms | numba 1-thread {timeit(lambda:nb_vision_st(pos,head,R,L,VR,nray,fov,tout2,hout2)):.2f} ms | numba parallel {timeit(lambda:nb_vision(pos,head,R,L,VR,nray,fov,tout,hout)):.2f} ms")
