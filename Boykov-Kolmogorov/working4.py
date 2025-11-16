from PIL import Image
import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt

MAX_SIDE = 256      # max image side to work on
K_CLUSTERS = 3      # k-means clusters for auto-seeding
LAMBDA = 5.0       # pairwise strength
SIGMA = 5.0        # color similarity scale (pairwise)
STRONG_SEED = 1e6   # capacity to enforce hard seeds

def fast_kmeans(pixels, k=3, iters=20):
    # pixels: (N,3) float array
    N = pixels.shape[0]
    idx = np.random.choice(N, k, replace=False)
    centers = pixels[idx].astype(float)
    for _ in range(iters):
        # vectorized assignment
        d2 = np.sum((pixels[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,k)
        labels = np.argmin(d2, axis=1)
        any_empty = False
        for j in range(k):
            members = pixels[labels == j]
            if len(members) == 0:
                # reinit empty center
                centers[j] = pixels[np.random.randint(0, N)]
                any_empty = True
            else:
                centers[j] = members.mean(axis=0)
        if not any_empty:
            pass
    return labels.reshape(-1), centers

class BoykovKolmogorov:
    def __init__(self, n):
        self.n = n
        # capacities stored as dict-of-dict for sparse graph: cap[u][v] = capacity (float)
        self.cap = [dict() for _ in range(n)]
        # flow[u][v] stored only when nonzero; we'll keep symmetric negative representation for reverse
        self.flow = [dict() for _ in range(n)]
        self.adj = [[] for _ in range(n)]
    def add_edge(self, u, v, c):
        if c <= 0:
            return
        # sum capacities if edge already exists
        self.cap[u][v] = self.cap[u].get(v, 0.0) + float(c)
        # ensure reverse present as key (for adjacency traversal)
        if v not in self.cap[v]:
            # do not set capacity of reverse; keep as absent or zero
            # leave cap[v][u] undefined unless added explicitly
            pass
        # adjacency (allow duplicates prevented)
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
    def residual(self, u, v):
        # residual of u->v = cap[u][v] - flow[u][v] (if cap exists) + (flow[v].get(u,0) if reverse flow exists)
        forward = self.cap[u].get(v, 0.0) - self.flow[u].get(v, 0.0)
        # reverse residual = self.flow[v].get(u, 0.0) (amount can be pushed back)
        reverse = self.flow[v].get(u, 0.0)
        return forward + reverse
    def _augment(self, path_edges, bott):
        # path_edges: list of (a,b) edges in direction of augmentation
        for a,b in path_edges:
            # if forward capacity exists, we increase flow[a][b]
            if self.cap[a].get(b, 0.0) > 0:
                self.flow[a][b] = self.flow[a].get(b, 0.0) + bott
            else:
                # otherwise we are pushing back along reverse: decrease flow[b][a]
                self.flow[b][a] = self.flow[b].get(a, 0.0) - bott
    def max_flow(self, s, t):
        n = self.n
        label = [0]*n       # 1 = S-tree, -1 = T-tree, 0 = free
        parent = [None]*n   # (parent_node, (parent, node))
        active = deque()
        orphan_q = deque()
        # initialize
        label[s] = 1; label[t] = -1
        parent[s] = ("ROOT", None); parent[t] = ("ROOT", None)
        active.append(s); active.append(t)
        total = 0.0
        # helper to reconstruct path edges from node to root
        def path_to_root(x):
            edges = []
            while parent[x] and parent[x][0] != "ROOT":
                p, e = parent[x]
                edges.append(e)
                x = p
            return list(reversed(edges))
        while True:
            meeting = None
            # Growth
            while active and meeting is None:
                p = active.popleft()
                for q in self.adj[p]:
                    if self.residual(p, q) <= 1e-12:
                        continue
                    if label[q] == 0:
                        label[q] = label[p]
                        parent[q] = (p, (p,q))
                        active.append(q)
                    elif label[q] == -label[p]:
                        # found meeting edge
                        if label[p] == 1:
                            meeting = (p, q)
                        else:
                            meeting = (q, p)
                        break
            if meeting is None:
                break
            u, v = meeting
            path_s = path_to_root(u)
            path_t = path_to_root(v)
            aug_edges = path_s + [(u, v)] + [(b, a) for (a,b) in path_t[::-1]]
            bott = float('inf')
            for a,b in aug_edges:
                r = self.residual(a,b)
                if r < bott:
                    bott = r
            if bott <= 0 or bott == float('inf'):
                break
            self._augment(aug_edges, bott)
            total += bott
            orphan_q.clear()
            for a,b in aug_edges:
                if abs(self.residual(a,b)) < 1e-12:
                    if parent[b] and parent[b][0] == a:
                        orphan_q.append(b); parent[b] = None
                    if parent[a] and parent[a][0] == b:
                        orphan_q.append(a); parent[a] = None
            while orphan_q:
                o = orphan_q.popleft()
                tree = label[o]
                new_parent = None
                for nb in self.adj[o]:
                    if label[nb] != tree:
                        continue
                    if self.residual(nb, o) <= 1e-12:
                        continue
                    cur = nb
                    valid = True
                    visited_chk = set()
                    while cur != s and cur != t:
                        if parent[cur] is None:
                            valid = False
                            break
                        cur = parent[cur][0]
                        if cur in visited_chk:
                            valid = False
                            break
                        visited_chk.add(cur)
                    if valid:
                        new_parent = nb
                        break
                if new_parent is not None:
                    parent[o] = (new_parent, (new_parent, o))
                else:
                    label[o] = 0
                    for child in self.adj[o]:
                        if parent[child] and parent[child][0] == o:
                            orphan_q.append(child)
                            parent[child] = None
        visited = [False]*n
        q = deque([s]); visited[s] = True
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if not visited[v] and self.residual(u,v) > 1e-12:
                    visited[v] = True
                    q.append(v)
        return total, visited

def segment_image_with_bk(img_arr, auto_seed=True, fg_clusters=None, bg_cluster=None,
                          max_side=MAX_SIDE, lam=LAMBDA, sigma=SIGMA):
    H0, W0 = img_arr.shape[:2]
    scale = min(1.0, max_side / max(H0, W0))
    if scale < 1.0:
        new_size = (int(W0*scale), int(H0*scale))
        img_small = np.array(Image.fromarray(img_arr).resize(new_size, Image.LANCZOS))
    else:
        img_small = img_arr.copy()
    H, W = img_small.shape[:2]
    pixels = img_small.reshape(-1, 3).astype(float)
    Npix = H * W
    SOURCE = 0; SINK = 1
    N = 2 + Npix
    if auto_seed:
        labels_flat, centers = fast_kmeans(pixels, k=K_CLUSTERS, iters=25)
        labels2d = labels_flat.reshape(H, W)
        brightness = centers.mean(axis=1)
        order = np.argsort(brightness)
        bg_cluster_choice = order[-1]
        center = np.array([H/2, W/2])
        spatial_centroids = []
        for k in range(K_CLUSTERS):
            ys, xs = np.where(labels2d == k)
            if len(ys) == 0:
                spatial_centroids.append(np.array([H/2, W/2]))
            else:
                spatial_centroids.append(np.array([ys.mean(), xs.mean()]))
        dists = [np.linalg.norm(spatial_centroids[k] - center) for k in range(K_CLUSTERS)]
        candidate = [k for k in range(K_CLUSTERS) if k != bg_cluster_choice]
        candidate_sorted = sorted(candidate, key=lambda k: dists[k])
        fg_clusters_choice = candidate_sorted[:1]  # one cluster as fg seed
        fg_clusters_choice = list(fg_clusters_choice)
    else:
        labels2d = None
        bg_cluster_choice = bg_cluster
        fg_clusters_choice = fg_clusters if fg_clusters else []
    bk = BoykovKolmogorov(N)
    def nid(i,j): return 2 + i*W + j
    if auto_seed:
        fg_mask = np.isin(labels2d, fg_clusters_choice)
        bg_mask = (labels2d == bg_cluster_choice)
        if fg_mask.sum() == 0:
            fg_color = np.array([255.,255.,255.])
        else:
            fg_color = pixels[fg_mask.reshape(-1)].mean(axis=0)
        if bg_mask.sum() == 0:
            bg_color = np.array([0.,0.,0.])
        else:
            bg_color = pixels[bg_mask.reshape(-1)].mean(axis=0)
    else:
        fg_color = np.array([255.,255.,255.])
        bg_color = np.array([0.,0.,0.])
    eps = 1e-6
    for i in range(H):
        for j in range(W):
            pid = i*W + j
            node = nid(i,j)
            col = pixels[pid]
            dfg = np.linalg.norm(col - fg_color)
            dbg = np.linalg.norm(col - bg_color)
            aff_fg = 1.0 / (dfg + eps)
            aff_bg = 1.0 / (dbg + eps)
            cap_fg = aff_fg * 100.0
            cap_bg = aff_bg * 100.0
            if auto_seed:
                if labels2d[i,j] in fg_clusters_choice:
                    cap_fg = STRONG_SEED; cap_bg = 0.0
                if labels2d[i,j] == bg_cluster_choice:
                    cap_bg = STRONG_SEED; cap_fg = 0.0
            bk.add_edge(SOURCE, node, cap_fg)
            bk.add_edge(node, SINK, cap_bg)
    two_sigma_sq = 2 * (sigma ** 2)
    for i in range(H):
        for j in range(W):
            u = nid(i,j)
            for di,dj in ((1,0),(0,1)):
                ni, nj = i+di, j+dj
                if ni < H and nj < W:
                    v = nid(ni,nj)
                    diff = pixels[i*W+j] - pixels[ni*W+nj]
                    diff_norm = np.linalg.norm(diff)
                    w = lam * math.exp(- (diff_norm*diff_norm) / two_sigma_sq)
                    if w > 1e-8:
                        bk.add_edge(u, v, w)
                        bk.add_edge(v, u, w)
    print("Running BK on image ({}x{} pixels)...".format(H, W))
    flow, S_reach = bk.max_flow(SOURCE, SINK)
    print("BK done. flow=", flow)
    mask_small = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if S_reach[nid(i,j)]:
                mask_small[i,j] = 255
    if scale < 1.0:
        mask = np.array(Image.fromarray(mask_small).resize((W0, H0), Image.NEAREST))
    else:
        mask = mask_small
    return mask

def main():
    input_path = "path/to/image.png"
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    mask = segment_image_with_bk(arr, auto_seed=True)
    # save and show
    outpath = "seg_mask.png"
    Image.fromarray(mask).save(outpath)
    print("Saved segmentation mask to", outpath)
    # Show overlay
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(arr); plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Segmentation mask (white = foreground)")
    plt.imshow(mask, cmap='gray'); plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
