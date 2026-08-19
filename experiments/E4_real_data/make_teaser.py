"""Figure 1: pipeline overview on the real cyclo-octane data."""
import os, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cyclooctane as co
from codim_sweep import geodesic_landmark_cover

CACHE = 'teaser_cache.npz'

def data():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z['emb'], z['lab'], z['sing'], z['charts']
    X = co.load_cyclooctane('pointsCycloOctane.mat')
    from sklearn.manifold import Isomap
    emb_all = Isomap(n_neighbors=10, n_components=3).fit_transform(X)
    Xr, keep = co.remove_singularities(X, dilate=0.3)
    st = co.stratify(Xr, min_cluster_size=150)
    lab = -np.ones(len(X), int)                 # -1 = singular/removed
    idx_keep = np.where(keep)[0]
    for c, s in enumerate(st):
        lab[idx_keep[s]] = c
    P = Xr[st[0]]; Pn = P - P.mean(0); Pn /= np.sqrt((Pn**2).sum(1).mean())
    A = geodesic_landmark_cover(Pn, max(8, round(len(Pn)/180)), percentile=8, seed=42)
    ch = -np.ones(len(X), int)
    for c, a in enumerate(A):
        ch[idx_keep[st[0]][a]] = c              # last chart wins on overlaps
    np.savez(CACHE, emb=emb_all, lab=lab, sing=~keep, charts=ch)
    return emb_all, lab, ~keep, ch


def panel(ax, emb, colours, title, sizes=3.2, alpha=.55):
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], s=sizes, c=colours, alpha=alpha,
               linewidths=0, depthshade=False)
    ax.set_title(title, fontsize=8.5, pad=-2)
    ax.view_init(elev=18, azim=45)
    ax.set_axis_off()                      # no panes/ticks: maximise the data
    lo, hi = emb.min(0), emb.max(0)        # tight, equal-aspect limits
    c, r = (lo+hi)/2, (hi-lo).max()/2*0.62
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)


def main(out='cyclo_fig/teaser.png'):
    emb, lab, sing, ch = data()
    fig = plt.figure(figsize=(9.2, 2.5))
    KL, SP, GR = '#d62728', '#1f77b4', '#c7c7c7'
    # (a) raw
    ax = fig.add_subplot(141, projection='3d')
    panel(ax, emb, GR, '(a) point cloud: $6040$ points in $\\mathbb{R}^{24}$')
    # (b) strata
    ax = fig.add_subplot(142, projection='3d')
    col = np.where(lab < 0, GR, np.where(lab == 0, KL, SP))
    panel(ax, emb, col, '(b) strata: Klein (red), spheres (blue)')
    # (c) charts on the Klein stratum
    ax = fig.add_subplot(143, projection='3d')
    cmap = plt.get_cmap('turbo')
    nch = ch.max() + 1
    col = np.array([GR]*len(emb), dtype=object)
    m = ch >= 0
    col[m] = [matplotlib.colors.to_hex(cmap(c/max(nch-1, 1))) for c in ch[m]]
    panel(ax, emb, list(col), f'(c) atlas: {nch} chart autoencoders')
    # (d) payoff: the sign cocycle on the nerve of the Klein stratum
    ax = fig.add_subplot(144)
    nerve = 'nerve_figs/nerve_klein.png'
    if os.path.exists(nerve):
        im = plt.imread(nerve)
        h, w = im.shape[:2]
        ax.imshow(im[int(.08*h):int(.99*h), int(.02*w):int(.98*w)])
    ax.set_axis_off()
    ax.set_title('(d) sign cocycle on the nerve', fontsize=8.5, pad=-2)
    fig.text(0.5, 0.015, 'certified: Klein stratum $w_1\\neq0$ (non-orientable);'
             ' the three sphere strata $w_1=0$ (orientable)', ha='center', fontsize=8)
    fig.subplots_adjust(left=.005, right=.995, top=.95, bottom=.10, wspace=.0)
    fig.savefig(out, dpi=190, bbox_inches='tight')
    print('saved', out)


if __name__ == '__main__':
    os.makedirs('cyclo_fig', exist_ok=True); main()
