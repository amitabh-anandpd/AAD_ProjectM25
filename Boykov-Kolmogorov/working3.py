from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import math
import random
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Load and resize ----------------
path = "path/to/image.jpg"  # replace with your image path
img = Image.open(path).convert("RGB")
# resize to speed up
scale = 512 / max(img.size)
new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
img = img.resize(new_size, Image.LANCZOS)
arr = np.array(img)
H, W = arr.shape[:2]

# ---------------- Simple k-means on Lab (approx via RGB->Luminance) ----------------
# We'll use a 4-cluster kmeans on RGB values reshaped.
pixels = arr.reshape(-1, 3).astype(float)

def kmeans(X, k=4, iters=20):
    # initialize: sample k pixels
    idx = np.random.choice(len(X), k, replace=False)
    centers = X[idx].astype(float)
    for _ in range(iters):
        # assign
        dists = np.sum((X[:,None,:] - centers[None,:,:])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        # if any empty, reinit randomly
        for i in range(k):
            if not np.any(labels==i):
                new_centers[i] = X[np.random.choice(len(X))]
        centers = new_centers
    return labels.reshape(H,W), centers

labels, centers = kmeans(pixels, k=4, iters=30)

# compute brightness (L channel approx) for centers
brightness = centers.mean(axis=1)  # mean RGB as proxy for luminance
order = np.argsort(brightness)  # low to high

# identify background cluster: the cluster corresponding to bright outer white (highest brightness)
bg_cluster = order[-1]

# Identify white-king cluster: cluster with very high brightness but not background (maybe 2nd highest)
white_cluster = order[-2]

# Identify black-king cluster: cluster with lowest brightness but spatially near center
black_candidate = order[0]

# However background circle is dark gradient, so black cluster might mix with background.
# To refine: pick cluster whose centroid spatial location is near image center for black king.
centroids_spatial = []
for k in range(4):
    ys, xs = np.where(labels==k)
    if len(ys)>0:
        centroids_spatial.append((ys.mean(), xs.mean()))
    else:
        centroids_spatial.append((H/2, W/2))
# distance to image center
center = (H/2, W/2)
dist_to_center = [math.hypot(cs[0]-center[0], cs[1]-center[1]) for cs in centroids_spatial]
# black king cluster = cluster with fairly low brightness but small distance to center (prefer)
candidates = list(range(4))
candidates_sorted = sorted(candidates, key=lambda k: (brightness[k], dist_to_center[k]))
black_cluster = candidates_sorted[0]

# Ensure white_cluster != bg_cluster; if equal pick next
if white_cluster == bg_cluster:
    white_cluster = order[-3]

# visualize clusters and chosen clusters
fig, axs = plt.subplots(1,5, figsize=(14,3))
axs[0].imshow(arr); axs[0].set_title("Input"); axs[0].axis('off')
for i in range(4):
    mask = (labels==i)
    display = arr.copy()
    display[~mask] = 255
    axs[i+1].imshow(display); axs[i+1].set_title(f"Cluster {i}\nbright={brightness[i]:.1f}\ncenter dist={dist_to_center[i]:.1f}")
    axs[i+1].axis('off')
plt.suptitle(f"Chosen clusters -> bg: {bg_cluster}, white: {white_cluster}, black: {black_cluster}")
plt.show()

# ---------------- Build BK implementation ----------------
class BoykovKolmogorov:
    def __init__(self, n):
        self.n = n
        self.cap = [defaultdict(float) for _ in range(n)]
        self.flow = [defaultdict(float) for _ in range(n)]
        self.adj = [set() for _ in range(n)]
    def add_edge(self,u,v,c):
        if c<=0: return
        self.cap[u][v] += float(c)
        self.cap[v][u] += 0.0
        self.adj[u].add(v); self.adj[v].add(u)
    def residual(self,u,v):
        return self.cap[u].get(v,0.0) - self.flow[u].get(v,0.0)
    def max_flow(self,s,t):
        n=self.n
        label=[0]*n; parent=[None]*n; active=deque(); orphan=deque()
        label[s]=1; label[t]=-1; parent[s]=("root",None); parent[t]=("root",None)
        active.extend([s,t])
        total=0.0
        def path_to_root(x):
            path=[]
            while parent[x] and parent[x][0]!="root":
                p,edge=parent[x]; path.append(edge); x=p
            return list(reversed(path))
        while True:
            meeting=None
            while active and meeting is None:
                p=active.popleft()
                for q in list(self.adj[p]):
                    if self.residual(p,q)<=1e-9: continue
                    if label[q]==0:
                        label[q]=label[p]; parent[q]=(p,(p,q)); active.append(q)
                    elif label[q]==-label[p]:
                        if label[p]==1: meeting=(p,q)
                        else: meeting=(q,p)
                        break
            if meeting is None: break
            u,v=meeting
            ps=path_to_root(u); pt=path_to_root(v)
            aug = ps + [(u,v)] + [(b,a) for (a,b) in pt[::-1]]
            bott=float("inf")
            for a,b in aug:
                bott=min(bott, self.residual(a,b))
            for a,b in aug:
                self.flow[a][b]=self.flow[a].get(b,0.0)+bott
                self.flow[b][a]=-self.flow[a][b]
            total += bott
            orphan.clear()
            for a,b in aug:
                if abs(self.residual(a,b))<1e-9:
                    if parent[b] and parent[b][0]==a:
                        orphan.append(b); parent[b]=None
                    if parent[a] and parent[a][0]==b:
                        orphan.append(a); parent[a]=None
            while orphan:
                o=orphan.popleft(); tree=label[o]; newp=None
                for q in list(self.adj[o]):
                    if label[q]!=tree or self.residual(q,o)<=1e-9: continue
                    v2=q; valid=True
                    while v2!=s and v2!=t:
                        if parent[v2] is None: valid=False; break
                        v2 = parent[v2][0]
                        if v2=="root": break
                    if valid:
                        newp=q; break
                if newp:
                    parent[o]=(newp,(newp,o))
                else:
                    label[o]=0
                    for q in list(self.adj[o]):
                        if parent[q] and parent[q][0]==o:
                            orphan.append(q); parent[q]=None
        # residual reachability
        vis=[False]*n; q=deque([s]); vis[s]=True
        while q:
            u=q.popleft()
            for v in list(self.adj[u]):
                if not vis[v] and self.residual(u,v)>1e-9:
                    vis[v]=True; q.append(v)
        return total, vis

# ---------------- Helper: do segmentation given foreground clusters list ----------------
def run_segmentation(fg_clusters):
    # build graph
    N = 2 + H*W
    SOURCE, SINK = 0, 1
    g = BoykovKolmogorov(N)
    # compute cluster centroids in color space for costs
    fg_pixels = np.isin(labels, fg_clusters)
    bg_pixels = (labels == bg_cluster)
    fg_color = pixels[fg_pixels.reshape(-1)].mean(axis=0) if np.any(fg_pixels) else np.array([255,255,255])
    bg_color = pixels[bg_pixels.reshape(-1)].mean(axis=0) if np.any(bg_pixels) else np.array([255,255,255])
    # parameters
    lam = 60.0
    sigma = 15.0
    # add unary capacities: based on color distance to fg/bg centroids
    for i in range(H):
        for j in range(W):
            nid = 2 + i*W + j
            col = pixels[i*W+j]
            # cost: distance to fg centroid (smaller -> more likely fg) and to bg centroid
            d_fg = np.linalg.norm(col - fg_color)
            d_bg = np.linalg.norm(col - bg_color)
            # map to capacities: larger capacity to source means more likely fg
            # use softmax-like mapping
            eps = 1e-6
            # invert distances to affinities
            aff_fg = 1.0 / (d_fg + eps)
            aff_bg = 1.0 / (d_bg + eps)
            # scale to capacities
            cap_fg = aff_fg * 100.0
            cap_bg = aff_bg * 100.0
            # strong seeds: if pixel belongs to fg_clusters, make it strong FG; if bg_cluster, strong BG
            if labels[i,j] in fg_clusters:
                cap_fg = 1e6; cap_bg = 0.0
            if labels[i,j] == bg_cluster:
                cap_bg = 1e6; cap_fg = 0.0
            g.add_edge(SOURCE, nid, cap_fg)
            g.add_edge(nid, SINK, cap_bg)
    # pairwise edges (4-neighborhood)
    for i in range(H):
        for j in range(W):
            u = 2 + i*W + j
            for di,dj in [(1,0),(0,1)]:
                ni, nj = i+di, j+dj
                if ni < H and nj < W:
                    v = 2 + ni*W + nj
                    diff = np.linalg.norm(pixels[i*W+j] - pixels[ni*W+nj])
                    w = lam * math.exp(-(diff*diff)/(2*(sigma*sigma)))
                    g.add_edge(u, v, w)
                    g.add_edge(v, u, w)
    flow, S_reach = g.max_flow(SOURCE, SINK)
    mask = np.zeros((H,W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if S_reach[2 + i*W + j]:
                mask[i,j]=255
    return flow, mask

# ---------------- Run options A, B, C ----------------
# Option A: both kings -> choose clusters that are not background; take two clusters with nearest centroids to center
# We'll pick clusters whose spatial centroid is within 0.6*max_dist from image center (likely kings)
dist_thresh = max(H,W)*0.6
spatial_dists = np.array(dist_to_center)
candidate_clusters = [i for i in range(4) if spatial_dists[i] < dist_thresh and i != bg_cluster]
# If less than 2, include next brightest/ darkest
candidate_clusters = list(dict.fromkeys(candidate_clusters))  # unique
if len(candidate_clusters) < 2:
    # pick top two by brightness that are not bg
    candidate_clusters = [c for c in order[::-1] if c!=bg_cluster][:2]
optA_fg = candidate_clusters[:2]
# Option B: black king only -> cluster with lowest brightness near center
optB_fg = [black_cluster]
# Option C: white king only -> white_cluster
optC_fg = [white_cluster]

print("Clusters chosen: bg:", bg_cluster, "white:", white_cluster, "black:", black_cluster)
print("Option A fg clusters:", optA_fg, "Option B:", optB_fg, "Option C:", optC_fg)

flowA, maskA = run_segmentation(optA_fg)
flowB, maskB = run_segmentation(optB_fg)
flowC, maskC = run_segmentation(optC_fg)

# Display results
fig, axes = plt.subplots(2,4, figsize=(12,6))
axes[0,0].imshow(arr); axes[0,0].set_title("Input"); axes[0,0].axis('off')
axes[0,1].imshow(maskA, cmap='gray'); axes[0,1].set_title(f"Option A: both kings\nflow={flowA:.1f}"); axes[0,1].axis('off')
axes[0,2].imshow(maskB, cmap='gray'); axes[0,2].set_title(f"Option B: black king\nflow={flowB:.1f}"); axes[0,2].axis('off')
axes[0,3].imshow(maskC, cmap='gray'); axes[0,3].set_title(f"Option C: white king\nflow={flowC:.1f}"); axes[0,3].axis('off')

# overlay masks on image
def overlay(img_arr, mask, color):
    out = img_arr.copy().astype(float)/255.0
    col = np.array(color)/255.0
    alpha = 0.6
    out[mask==255] = (1-alpha)*out[mask==255] + alpha*col
    return (out*255).astype(np.uint8)

axes[1,0].imshow(arr); axes[1,0].set_title("Input"); axes[1,0].axis('off')
axes[1,1].imshow(overlay(arr, maskA, (255,0,0))); axes[1,1].set_title("A overlay"); axes[1,1].axis('off')
axes[1,2].imshow(overlay(arr, maskB, (0,255,0))); axes[1,2].set_title("B overlay"); axes[1,2].axis('off')
axes[1,3].imshow(overlay(arr, maskC, (0,0,255))); axes[1,3].set_title("C overlay"); axes[1,3].axis('off')

plt.tight_layout()
plt.show()

# Save masks to /mnt/data so user can download if needed
from PIL import Image as PILImage
PILImage.fromarray(maskA).save("mask_optionA.png")
PILImage.fromarray(maskB).save("mask_optionB.png")
PILImage.fromarray(maskC).save("mask_optionC.png")