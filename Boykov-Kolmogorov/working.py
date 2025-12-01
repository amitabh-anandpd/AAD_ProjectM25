# Implementation of a simplified Boykov-Kolmogorov max-flow algorithm.

from collections import deque, defaultdict

class BoykovKolmogorov:
    def __init__(self, n):
        self.n = n
        # capacities: dict of dicts for sparse representation
        self.cap = [defaultdict(int) for _ in range(n)]
        # residual capacities stored implicitly via cap - flow
        self.flow = [defaultdict(int) for _ in range(n)]
        # adjacency list for iteration
        self.adj = [set() for _ in range(n)]
    
    def add_edge(self, u, v, capacity):
        """Add directed edge u->v with given capacity. If edge exists, capacities are summed."""
        if capacity <= 0:
            return
        self.cap[u][v] += capacity
        self.adj[u].add(v)
        self.adj[v].add(u)  # ensure reverse exists for residual traversal
    
    def residual_capacity(self, u, v):
        """Residual capacity from u to v = cap[u][v] - flow[u][v] + flow[v][u] (if reverse flow exists).
           For this sparse representation we treat reverse as separate entries and access safely."""
        # forward residual
        forward = self.cap[u].get(v, 0) - self.flow[u].get(v, 0)
        # reverse residual (flow that can be pushed back)
        reverse = self.flow[v].get(u, 0)
        return forward + reverse
    
    def _augment_path(self, path, bottleneck):
        """Augment flow along path (list of (u,v) edges) by bottleneck"""
        for u, v in path:
            # if forward edge exists (capacity present), increase flow
            if self.cap[u].get(v, 0) > 0:
                self.flow[u][v] = self.flow[u].get(v, 0) + bottleneck
            else:
                # otherwise we are pushing back along reverse direction: decrease flow on v->u
                self.flow[v][u] = self.flow[v].get(u, 0) - bottleneck
    
    def max_flow(self, s, t):
        """Compute max flow from s to t using a simplified BK algorithm.
           Returns (flow_value, partition_S) where partition_S is a boolean list marking nodes reachable from s
           in the final residual graph (i.e., the S side of the min-cut)."""
        n = self.n
        # node label: 1 => S-tree, -1 => T-tree, 0 => free
        label = [0]*n
        parent = [None]*n  # parent pointer (node, via_node) storing the parent node in tree (for path reconstruction)
        active = deque()   # active nodes to expand (nodes at frontier)
        
        # initialize trees
        label[s] = 1
        label[t] = -1
        parent[s] = ("ROOT", None)
        parent[t] = ("ROOT", None)
        active.append(s)
        active.append(t)
        
        # orphan queue for adoption phase
        orphan_queue = deque()
        
        total_flow = 0
        
        def find_path_meeting(u, v):
            """Build augmenting path when u (in S-tree) has neighbor v (in T-tree). 
               Return edge list path as sequence of (a,b) from s...t"""
            # reconstruct path s -> ... -> u
            path_s = []
            x = u
            while x != s:
                px, via = parent[x]
                path_s.append((px, x))
                x = px
            path_s.reverse()  # now s -> ... -> u
            # edge (u, v)
            path = path_s + [(u, v)]
            # reconstruct path v -> ... -> t
            y = v
            while y != t:
                py, via = parent[y]
                # note parent in T-tree points towards t, but edges go y->py for path to t
                path.append((y, py))
                y = py
            return path
        
        while True:
            # ---- Growth phase ----
            meeting = None  # (u,v) edge where trees meet
            while active and meeting is None:
                p = active.popleft()
                # iterate neighbors
                for q in list(self.adj[p]):
                    if self.residual_capacity(p, q) <= 0:
                        continue
                    if label[q] == 0:
                        # add q to same tree as p
                        label[q] = label[p]
                        parent[q] = (p, (p, q))
                        active.append(q)
                    elif label[q] == -label[p]:
                        # found connection between S-tree and T-tree
                        if label[p] == 1:
                            meeting = (p, q)
                        else:
                            meeting = (q, p)
                        break
            if meeting is None:
                # no augmenting path found -> finished
                break
            
            # ---- Augmentation phase ----
            u, v = meeting
            path = find_path_meeting(u, v)  # list of edges (a,b)
            # compute bottleneck
            bottleneck = float('inf')
            for a, b in path:
                res = self.residual_capacity(a, b)
                if res < bottleneck:
                    bottleneck = res
            if bottleneck == 0 or bottleneck == float('inf'):
                # shouldn't happen, but guard
                break
            # apply augmentation
            # before augmentation, record which edges become saturated to detect orphans
            saturated_edges = []
            for a, b in path:
                # compute forward capacity available (cap - flow)
                forward = self.cap[a].get(b, 0) - self.flow[a].get(b, 0)
                if forward >= bottleneck and self.cap[a].get(b, 0) > 0:
                    # forward augmentation on a->b
                    self.flow[a][b] = self.flow[a].get(b, 0) + bottleneck
                    if self.cap[a][b] - self.flow[a][b] == 0:
                        saturated_edges.append((a, b))
                else:
                    # either pushing on reverse or forward < bottleneck (use reverse flow decrease)
                    # if reverse flow exists, we decrease it
                    if self.flow[b].get(a, 0) > 0:
                        # decrease reverse flow
                        self.flow[b][a] = self.flow[b].get(a, 0) - bottleneck
                        if self.flow[b][a] == 0:
                            # removal of reverse flow may "saturate" nothing notable for parent structure
                            pass
                    else:
                        # if forward capacity was zero and reverse flow zero, this is unexpected
                        # skip
                        pass
            total_flow += bottleneck
            
            # determine orphans: nodes whose parent edge is now saturated (for nodes in both trees)
            orphan_queue.clear()
            for a, b in saturated_edges:
                # if b was child of a in tree, b becomes orphan
                # check parent pointers: parent[child] = (parent_node, via_edge)
                if parent[b] and parent[b][0] == a:
                    orphan_queue.append(b)
                # if a was child of b (possible in reverse orientation), handle similarly
                if parent[a] and parent[a][0] == b:
                    orphan_queue.append(a)
            
            # ---- Adoption phase: process orphans ----
            while orphan_queue:
                o = orphan_queue.popleft()
                # try to find new parent for o inside same tree
                tree_label = label[o]
                found_parent = False
                # search neighbors for a valid parent in same tree with residual capacity > 0
                for nb in self.adj[o]:
                    if label[nb] != tree_label:
                        continue
                    # check residual capacity from nb -> o (we need path from root to nb then nb->o)
                    if self.residual_capacity(nb, o) > 0:
                        # ensure nb has a valid path to tree root (i.e., parent chain leads to root)
                        cur = nb
                        valid_chain = True
                        visited_in_chain = set()
                        while cur != s and cur != t and cur != None and cur not in visited_in_chain:
                            visited_in_chain.add(cur)
                            pnode, _ = parent[cur]
                            if pnode == "ROOT" or pnode is None:
                                break
                            cur = pnode
                        # cur should have been root of same tree
                        # simplification: accept nb as parent if parent[nb] is set and nb is labeled
                        if parent[nb] is not None:
                            parent[o] = (nb, (nb, o))
                            found_parent = True
                            break
                if not found_parent:
                    # remove o from tree (it becomes free)
                    label[o] = 0
                    # its children become orphans
                    for child in range(n):
                        if parent[child] and parent[child][0] == o:
                            orphan_queue.append(child)
                            parent[child] = None
                    parent[o] = None
                    # do not enqueue o to active even if it loses status
        
            # after adoption, rebuild active queue with all frontier nodes of both trees
            active.clear()
            for i in range(n):
                if label[i] != 0:
                    # node is in some tree; if it has any residual outgoing edge, it's active for expansion
                    for nb in self.adj[i]:
                        if self.residual_capacity(i, nb) > 0 and label[nb] == 0:
                            active.append(i)
                            break
            
        # After algorithm finishes, compute S-side of final cut using reachable in residual graph
        visited = [False]*n
        q = deque([s])
        visited[s] = True
        while q:
            u = q.popleft()
            for v in self.adj[u]:
                if not visited[v] and self.residual_capacity(u, v) > 0:
                    visited[v] = True
                    q.append(v)
        
        return total_flow, visited

if __name__ == "__main__":
    # Build a small test graph (classic example)
    # Graph:
    # s(0) -> 1 (16), s -> 2 (13)
    # 1 -> 2 (10), 2 -> 1 (4)
    # 1 -> 3 (12), 2 -> 4 (14)
    # 3 -> 2 (9), 4 -> 3 (7)
    # 3 -> t(5) (20), 4 -> t (4)
    g = BoykovKolmogorov(6)
    s, t = 0, 5
    g.add_edge(0,1,16)
    g.add_edge(0,2,13)
    g.add_edge(1,2,10)
    g.add_edge(2,1,4)
    g.add_edge(1,3,12)
    g.add_edge(2,4,14)
    g.add_edge(3,2,9)
    g.add_edge(4,3,7)
    g.add_edge(3,5,20)
    g.add_edge(4,5,4)
    
    maxflow, S_partition = g.max_flow(s, t)
    print("Max flow computed:", maxflow)
    print("Nodes in source side of min-cut (S):", [i for i,v in enumerate(S_partition) if v])
    print("Nodes in sink side of min-cut (T):", [i for i,v in enumerate(S_partition) if not v])
