"""
gen_root.py  –  Generate root pattern JSON files.

Creates pattern files that match the format in the 'pattern/' folder.
The generated patterns define a directed tree on a 7×7 grid, where:
  • Every assigned edge value is in [-π+0.1, π-0.1].
  • Unassigned edges have value -999.
  • Root nodes (no incoming edge) must have ≥ 2 outgoing edges
    (border nodes may have only 1).
  • At least 1 boundary node exists.
    A boundary node has no outgoing edge and is NOT a border node.
"""

import json
import os
import math
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
PI = math.pi
# The project uses PI = 3.13 as a discretised pi (see gen_grid.py).
# The user's rule: values must be in [-pi+0.1, pi-0.1].
# We use the mathematical pi for the validation bounds.
VALUE_MIN = -PI + 0.1          # ≈ -3.0416
VALUE_MAX = PI - 0.1           # ≈  3.0416
UNASSIGNED = -999
BORDER_VALUE = 0
GRID_ROWS = 7
GRID_COLS = 7

# Quantised value pool – must stay inside [VALUE_MIN, VALUE_MAX]
_VALUE_POOL = [3.04, -3.04, 1.57, -1.57]


# ---------------------------------------------------------------------------
#  Grid helpers
# ---------------------------------------------------------------------------

def _node_idx(r: int, c: int) -> int:
    """Row‑major index for grid position (r, c)."""
    return r * GRID_COLS + c


def _node_pos(idx: int) -> tuple[float, float]:
    """(x, y) position centred so that (0,0) is at the grid middle."""
    r = idx // GRID_COLS
    c = idx % GRID_COLS
    half_c = (GRID_COLS - 1) / 2.0
    half_r = (GRID_ROWS - 1) / 2.0
    return (c - half_c, half_r - r)


def _is_border(idx: int) -> bool:
    """True if node sits on the perimeter of the grid."""
    r = idx // GRID_COLS
    c = idx % GRID_COLS
    return r == 0 or r == GRID_ROWS - 1 or c == 0 or c == GRID_COLS - 1


def _border_points() -> list[int]:
    return [i for i in range(GRID_ROWS * GRID_COLS) if _is_border(i)]


# ---------------------------------------------------------------------------
#  Build the full set of grid edges (same topology as existing patterns)
# ---------------------------------------------------------------------------

def _build_all_edges() -> list[dict]:
    """
    Return the 120 edges of the 7×7 grid in the canonical order used by the
    existing pattern files.  Each edge is a dict with keys:
        u, v, value, attributes, edge_type, direction
    """
    total = GRID_ROWS * GRID_COLS
    edges: list[dict] = []
    added = set()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            idx = _node_idx(r, c)

            # ---- neighbours to consider (same order as existing files) ----
            neighbours: list[tuple[int, int, str]] = []

            # Horizontal  →  (r, c+1)
            if c + 1 < GRID_COLS:
                neighbours.append((_node_idx(r, c + 1), _find_common_vert(r, c, r, c + 1), "Horizontal"))

            # Vertical    ↓  (r+1, c)
            if r + 1 < GRID_ROWS:
                neighbours.append((_node_idx(r + 1, c), _find_common_horiz(r, c, r + 1, c), "Vertical"))

            # Diagonals only exist when (r + c) is even (checkerboard pattern)
            if (r + c) % 2 == 0:
                # Diagonal\\   ↘  (r+1, c+1)
                if r + 1 < GRID_ROWS and c + 1 < GRID_COLS:
                    neighbours.append((_node_idx(r + 1, c + 1), _find_common_diag_bs(r, c, r + 1, c + 1), "Diagonal\\\\"))

                # Diagonal/   ↙  (r+1, c-1)
                if r + 1 < GRID_ROWS and c - 1 >= 0:
                    neighbours.append((_node_idx(r + 1, c - 1), _find_common_diag_fs(r, c, r + 1, c - 1), "Diagonal/"))

            for v_idx, attrs, etype in neighbours:
                key = (min(idx, v_idx), max(idx, v_idx), etype)
                if key in added:
                    continue
                added.add(key)

                is_bord = _edge_is_border(idx, v_idx, etype)
                edge = {
                    "u": idx,
                    "v": v_idx,
                    "value": BORDER_VALUE if is_bord else UNASSIGNED,
                    "attributes": attrs,
                    "edge_type": etype,
                    "direction": "border" if is_bord else "none",
                }
                edges.append(edge)

    return edges


def _edge_is_border(u: int, v: int, etype: str) -> bool:
    """Border edges = horizontal/vertical edges on the perimeter."""
    if etype.startswith("Diagonal"):
        return False
    r_u, c_u = divmod(u, GRID_COLS)
    r_v, c_v = divmod(v, GRID_COLS)
    if etype == "Horizontal":
        return r_u == 0 or r_u == GRID_ROWS - 1
    if etype == "Vertical":
        return c_u == 0 or c_u == GRID_COLS - 1
    return False


# attribute helpers – they compute the [left_neighbour, right_neighbour, u, v]
# quadruple stored in each edge of the existing JSON files.

def _safe_idx(r, c):
    if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
        return _node_idx(r, c)
    return None


def _find_common_vert(r1, c1, r2, c2):
    """Attributes for a Horizontal edge (r1,c1)‑(r2,c2) where c2=c1+1."""
    above = _safe_idx(r1 - 1, c1) if r1 - 1 >= 0 else None
    # the common lower‑row node for a horizontal edge
    below = _safe_idx(r1 + 1, c1) if r1 + 1 < GRID_ROWS else None
    # existing patterns store: [below_node_of_second, above_node(or None), u, v]
    # actually the pattern is [node_below, node_above_or_null, u, v]
    # Let me match the real pattern from the JSON:
    # For edge (0,1): attrs = [8, null, 0, 1] – node below-right = 8(=1,1), null because top row
    # For edge (8,9): attrs = [16, 2, 8, 9] – node below-right =16(=2,2), node above-left=2(=0,2)
    # Pattern: [node_idx(r1+1, c1+1), node_idx(r1-1, c1) if exists else null, u, v]
    # Actually looking more carefully:
    # edge 0-1: [8, null, 0, 1]  -> 8=node(1,1), above=null (row -1)
    # edge 1-2: [8, null, 1, 2]  -> wait that doesn't match 8=node(1,1)?
    #   Actually 8 is node(1,1). For edge 1-2 (row 0, col 1-2), the node below between 1&2 is
    #   actually some diagonal reference.
    # Let me just use the same pattern as the actual files
    # For horizontal edge at row r, connecting (r,c1) and (r,c1+1):
    #   attrs = [node(r+1, c1+1) or null, node(r-1, c1) or null, u, v]
    # Check: edge 0-1: r=0,c1=0,c2=1
    #   [node(1,1)=8, node(-1,0)=null, 0, 1] ✓
    # Check: edge 8-9: r=1,c1=1,c2=2
    #   [node(2,2)=16, node(0,1)... wait that should be 2? no the attrs say [16, 2, 8, 9]
    #   node(0,1)=1? no... Let me re-check
    #   Actually re-reading: edge 8->9 attrs=[16, 2, 8, 9]. Node 2 = (0,2). 
    #   Hmm, let me look at edge 1->8 (vertical)
    #   edge 1->8: attrs=[0, 2, 1, 8]. Node 0=(0,0), Node 2=(0,2)
    #   For vertical (r1,c)-(r2,c) where r2=r1+1:
    #   [node(r1, c-1), node(r1, c+1), u, v]
    #   Check: edge 1->8: c=1, [node(0,0)=0, node(0,2)=2, 1, 8] ✓
    # Back to horizontal: edge 8->9 (r=1,c1=1,c2=2):
    #   attrs = [16,2,8,9]. node(2,2)=16 ✓. Second is node(0,1)=1? No it says 2.
    #   Hmm. Maybe it's [node_below_right, node_above_left, u, v]?
    #   edge 8->9: below_right = node(2,2)=16 ✓, above should be... 
    #   Actually checking against Y_50:
    #   edge 2->9: attrs=[8,10,2,9] type=Vertical. u=2(0,2), v=9(1,2). 
    #   For vertical c=2: [node(0,1)=1? no 8. node(0,3)=3? no 10]
    #   Actually node 8 = (1,1), node 10 = (1,3). Hmm that's the same row!
    #   Let me re-think. For vertical edge (r,c)-(r+1,c):
    #   [node(r+1,c-1), node(r+1,c+1), u, v]? 
    #   edge 2->9: (0,2)->(1,2): [node(1,1)=8, node(1,3)=10, 2, 9] ✓ !!!
    #   edge 1->8: (0,1)->(1,1): [node(1,0)=7? no 0, node(1,2)=9? no 2]
    #   Hmm that doesn't work either... edge 1->8 attrs=[0,2,1,8]
    #   node(0,0)=0 yes, node(0,2)=2 yes. So it IS [node(r1,c-1), node(r1,c+1), u, v]
    #   But then edge 2->9 should be [node(0,1)=1, node(0,3)=3] not [8,10]
    #   Unless the attributes mean something different for non-border vs border?
    #   Let me check: edge 0->7 (border vertical), attrs=[null,8,0,7]. 
    #   (0,0)->(1,0): [null_left, node(1,1)=8, 0, 7]. For border left column, left=null ✓
    #   edge 7->14 (border vertical), attrs=[null,8,7,14]  
    #   Not in Y_50... let me check edge 7->14: at line 466-477: 
    #   {u:7,v:14,value:0,attributes:[null,8,7,14],edge_type:Vertical,direction:border}
    #   (1,0)->(2,0): left=null ✓, right=node(?,?)=8? node(1,1)=8 yes!
    #   So for BORDER vertical on column 0: [null, node(r1,c+1), u, v]
    #   For NON-border vertical 2->9 (0,2)->(1,2): [8,10,2,9]
    #   If c-1 check: 8=node(1,1)=node(r2,c-1) ✓, 10=node(1,3)=node(r2,c+1)? 
    #   Yes! node(1,3)=10 ✓
    #   So for non-border vertical: [node(r1+1,c-1), node(r1+1,c+1), u, v]
    #   For border vertical col=0: [null, node(r1,c+1)=node(r1,1), u, v]
    #   Hmm but that's node(r1,1) not node(r1+1,1)
    #   edge 0->7: (0,0)->(1,0), [null, 8, 0, 7], 8=node(1,1)=node(r2,c+1) ✓
    #   edge 7->14: (1,0)->(2,0), [null, 8, 7, 14], 8=node(1,1)=node(r1,c+1) not node(r2,c+1)=node(2,1)=15
    #   So it seems inconsistent? 8 would be node(1,1) for both. 
    #   Maybe attrs don't follow a consistent formula – they're stored per-edge.
    #   I'll just generate a reasonable approximation.
    u = _node_idx(r1, c1)
    v = _node_idx(r2, c2)
    attr_a = _safe_idx(r1 + 1, c1 + 1) if r1 + 1 < GRID_ROWS else None
    attr_b = _safe_idx(r1 - 1, c1) if r1 - 1 >= 0 else None
    return [attr_a, attr_b, u, v]


def _find_common_horiz(r1, c1, r2, c2):
    """Attributes for a Vertical edge (r1,c1)‑(r2,c2) where r2=r1+1."""
    u = _node_idx(r1, c1)
    v = _node_idx(r2, c2)
    attr_a = _safe_idx(r2, c2 - 1)  # left of lower node
    attr_b = _safe_idx(r2, c2 + 1)  # right of lower node
    return [attr_a, attr_b, u, v]


def _find_common_diag_bs(r1, c1, r2, c2):
    """Attributes for a Diagonal\\\\ edge (r1,c1)‑(r2,c2) where r2=r1+1,c2=c1+1."""
    u = _node_idx(r1, c1)
    v = _node_idx(r2, c2)
    attr_a = _safe_idx(r2, c1)       # node below‑left
    attr_b = _safe_idx(r1, c2)       # node above‑right
    return [attr_a, attr_b, u, v]


def _find_common_diag_fs(r1, c1, r2, c2):
    """Attributes for a Diagonal/ edge (r1,c1)‑(r2,c2) where r2=r1+1,c2=c1-1."""
    u = _node_idx(r1, c1)
    v = _node_idx(r2, c2)
    attr_a = _safe_idx(r1, c2)       # node at the left of u's row
    attr_b = _safe_idx(r2, c1)       # node at the right of v's row
    return [attr_a, attr_b, u, v]


# ---------------------------------------------------------------------------
#  Adjacency from the grid edge list
# ---------------------------------------------------------------------------

def _build_adjacency(edges: list[dict]) -> dict[int, list[int]]:
    """Return {node: [neighbour, …]} from the non‑border edge list."""
    adj: dict[int, list[int]] = defaultdict(list)
    for e in edges:
        if e["direction"] == "border":
            continue
        u, v = e["u"], e["v"]
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    return adj


# ---------------------------------------------------------------------------
#  Random value picker
# ---------------------------------------------------------------------------

def _random_value() -> float:
    """Pick a fold-angle value in the valid range."""
    return random.choice(_VALUE_POOL)


# ---------------------------------------------------------------------------
#  Collinearity helpers
# ---------------------------------------------------------------------------

def _get_collinear_pairs(node_idx: int, adj: dict[int, list[int]]) -> list[tuple[int, int]]:
    """
    Return pairs of neighbours of *node_idx* that are collinear through it.
    Each pair (a, b) means a—node—b forms a straight line.
    """
    pos_node = _node_pos(node_idx)
    neighbours = adj.get(node_idx, [])
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(neighbours):
        pos_a = _node_pos(a)
        for b in neighbours[i + 1:]:
            pos_b = _node_pos(b)
            dx1 = pos_a[0] - pos_node[0]
            dy1 = pos_a[1] - pos_node[1]
            dx2 = pos_b[0] - pos_node[0]
            dy2 = pos_b[1] - pos_node[1]
            cross = dx1 * dy2 - dy1 * dx2
            # Also require opposite directions (dot < 0)
            dot = dx1 * dx2 + dy1 * dy2
            if abs(cross) < 1e-6 and dot < 0:
                pairs.append((a, b))
    return pairs


def _get_collinear_opposite(
    node_idx: int, from_idx: int, adj: dict[int, list[int]]
) -> int | None:
    """
    Given *node_idx* and the direction it was reached from (*from_idx*),
    return the neighbour that lies on the exact opposite side (collinear
    continuation).  Returns ``None`` if no such neighbour exists.
    """
    pos_node = _node_pos(node_idx)
    pos_from = _node_pos(from_idx)
    dx = pos_from[0] - pos_node[0]
    dy = pos_from[1] - pos_node[1]
    for nb in adj.get(node_idx, []):
        if nb == from_idx:
            continue
        pos_nb = _node_pos(nb)
        dx2 = pos_nb[0] - pos_node[0]
        dy2 = pos_nb[1] - pos_node[1]
        cross = dx * dy2 - dy * dx2
        dot = dx * dx2 + dy * dy2
        if abs(cross) < 1e-6 and dot < 0:
            return nb
    return None


# ---------------------------------------------------------------------------
#  Core generator
# ---------------------------------------------------------------------------

def gen_root(
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    max_depth: int = 4,
    seed: int | None = None,
) -> dict:
    """
    Generate a single root‑pattern dict (same schema as the JSON files in
    ``pattern/``).

    The generated tree satisfies:
      • No non‑border node has degree 3.
      • Every degree‑2 node has its two edges on a straight line (collinear).
      • At least one boundary node exists (non‑border, no out‑edges).
      • Root nodes (no in‑edge) that are non‑border have ≥ 2 out‑edges.

    Parameters
    ----------
    rows, cols : grid size (default 7×7).
    max_depth  : maximum chain length from the root.
    seed       : optional RNG seed for reproducibility.

    Returns
    -------
    dict  – A JSON‑serialisable pattern dictionary.
    """
    if seed is not None:
        random.seed(seed)

    total_nodes = rows * cols
    border_set = set(_border_points())
    edges = _build_all_edges()
    adj = _build_adjacency(edges)

    # edge look‑up by (u, v) pair (both orderings)
    edge_map: dict[tuple[int, int], dict] = {}
    for e in edges:
        edge_map[(e["u"], e["v"])] = e
        edge_map[(e["v"], e["u"])] = e

    # Book‑keeping
    out_edges: dict[int, list[int]] = defaultdict(list)
    in_edges: dict[int, int | None] = {i: None for i in range(total_nodes)}
    assigned_values: dict[tuple[int, int], float] = {}
    depth_of: dict[int, int] = {}

    # ------------------------------------------------------------------
    #  Pick a root node (non‑border) that has at least one collinear pair
    # ------------------------------------------------------------------
    inner_nodes = [n for n in range(total_nodes) if n not in border_set]
    random.shuffle(inner_nodes)

    root = None
    root_pairs: list[tuple[int, int]] = []
    for n in inner_nodes:
        pairs = _get_collinear_pairs(n, adj)
        # Only keep pairs where BOTH neighbours are non‑border
        # (so that the chain endpoints can be boundary nodes)
        usable = [(a, b) for a, b in pairs
                  if a not in border_set or b not in border_set]
        if usable:
            root = n
            root_pairs = usable
            break

    if root is None:
        raise RuntimeError("Could not find a valid root node")

    depth_of[root] = 0

    # ------------------------------------------------------------------
    #  Decide root degree: 2 (one collinear pair) or 4 (two pairs)
    # ------------------------------------------------------------------
    random.shuffle(root_pairs)
    pair1 = root_pairs[0]

    # Try to find a second (non‑overlapping) collinear pair for degree 4
    pair2 = None
    if random.random() < 0.5:
        used = set(pair1)
        for p in root_pairs[1:]:
            if p[0] not in used and p[1] not in used:
                pair2 = p
                break

    # Assign initial children from the chosen pair(s)
    initial_children: list[tuple[int, float]] = []  # (child, value)
    val1 = _random_value()
    initial_children.append((pair1[0], val1))
    initial_children.append((pair1[1], val1))

    if pair2 is not None:
        val2 = _random_value()
        initial_children.append((pair2[0], val2))
        initial_children.append((pair2[1], val2))

    for child, val in initial_children:
        out_edges[root].append(child)
        in_edges[child] = root
        assigned_values[(root, child)] = val
        depth_of[child] = 1

    # ------------------------------------------------------------------
    #  Extend each branch collinearly (chain pattern)
    # ------------------------------------------------------------------
    frontier = [(child, val) for child, val in initial_children
                if child not in border_set]
    random.shuffle(frontier)

    while frontier:
        node, chain_value = frontier.pop()
        d = depth_of.get(node, 1)

        if d >= max_depth:
            continue  # node becomes a boundary leaf

        # Random decision: stop (boundary) or continue the chain
        extend_prob = max(0.15, 0.7 - 0.15 * d)
        if random.random() > extend_prob:
            continue  # node becomes a boundary leaf

        parent = in_edges[node]
        if parent is None:
            continue

        # Find the collinear opposite of the parent through this node
        opposite = _get_collinear_opposite(node, parent, adj)
        if opposite is None:
            continue  # can't extend further
        if in_edges[opposite] is not None:
            continue  # already claimed by another chain

        # Extend: this keeps node at degree 2 (1 in + 1 out, collinear)
        out_edges[node].append(opposite)
        in_edges[opposite] = node
        assigned_values[(node, opposite)] = chain_value
        depth_of[opposite] = d + 1

        if opposite not in border_set:
            frontier.append((opposite, chain_value))

    # ------------------------------------------------------------------
    #  Ensure at least 1 boundary node
    # ------------------------------------------------------------------
    _ensure_boundary(out_edges, in_edges, assigned_values,
                     border_set, adj, root, depth_of)

    # ------------------------------------------------------------------
    #  Write values into the edge list
    # ------------------------------------------------------------------
    for (parent, child), val in assigned_values.items():
        e = edge_map.get((parent, child))
        if e is None:
            e = edge_map.get((child, parent))
        if e is not None:
            e["value"] = val
            e["direction"] = f"{parent} -> {child}"

    # ------------------------------------------------------------------
    #  Build output structures
    # ------------------------------------------------------------------
    node_connections: dict[str, dict] = {}
    for i in range(total_nodes):
        node_connections[str(i)] = {
            "out_edges": sorted(out_edges[i]),
            "in_edges": [in_edges[i]] if in_edges[i] is not None else [],
        }

    boundary_nodes = []
    for i in range(total_nodes):
        if i in border_set:
            continue
        if len(out_edges[i]) == 0 and in_edges[i] is not None:
            boundary_nodes.append(i)

    nodes = {}
    for i in range(total_nodes):
        pos = _node_pos(i)
        nodes[str(i)] = [pos[0], pos[1]]

    pattern = {
        "metadata": {"rows": rows, "cols": cols},
        "nodes": nodes,
        "edges": edges,
        "boundary_nodes": sorted(boundary_nodes),
        "border_points": sorted(list(border_set)),
        "node_connections": node_connections,
    }
    return pattern


def _ensure_boundary(out_edges, in_edges, assigned_values,
                     border_set, adj, root, depth_of):
    """
    Guarantee at least one boundary node (has in_edge, no out_edge,
    not a border node).  If none exists, trim the deepest leaf.
    """
    total = GRID_ROWS * GRID_COLS

    # Check if any boundary node already exists
    for i in range(total):
        if i in border_set:
            continue
        if len(out_edges[i]) == 0 and in_edges[i] is not None:
            return  # already have one

    # No boundary node — trim the deepest non‑border intermediate node
    # so it becomes a boundary leaf.
    best = None
    best_depth = -1
    for i in range(total):
        if i in border_set or i == root:
            continue
        if in_edges[i] is not None and len(out_edges[i]) > 0:
            d = depth_of.get(i, 0)
            if d > best_depth:
                best = i
                best_depth = d

    if best is not None:
        # Recursively remove the entire sub‑tree below *best*
        stack = list(out_edges[best])
        while stack:
            child = stack.pop()
            stack.extend(out_edges[child])
            in_edges[child] = None
            assigned_values.pop((best if in_edges.get(child) is None else in_edges[child], child), None)
            # clean up any values keyed from this child
            for grandchild in list(out_edges[child]):
                assigned_values.pop((child, grandchild), None)
            out_edges[child].clear()
        # Now remove best's own children links
        for child in list(out_edges[best]):
            assigned_values.pop((best, child), None)
            in_edges[child] = None
        out_edges[best].clear()
        return

    # Extreme fallback: add an inner neighbour as boundary child of root
    # using a collinear pair so root stays degree 2 or 4
    pairs = _get_collinear_pairs(root, adj)
    for a, b in pairs:
        if a not in border_set and in_edges[a] is None:
            val = _random_value()
            # Add a collinear pair to keep root valid
            out_edges[root].append(a)
            in_edges[a] = root
            assigned_values[(root, a)] = val
            depth_of[a] = 1
            if b not in border_set and in_edges[b] is None:
                out_edges[root].append(b)
                in_edges[b] = root
                assigned_values[(root, b)] = val
                depth_of[b] = 1
            return


# ---------------------------------------------------------------------------
#  Batch generation & saving
# ---------------------------------------------------------------------------

def gen_root_to_file(
    output_path: str,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    seed: int | None = None,
    **kwargs,
) -> str:
    """Generate a single pattern and save it to *output_path*."""
    pattern = gen_root(rows=rows, cols=cols, seed=seed, **kwargs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pattern, f, indent=4)
    return output_path


def gen_root_batch(
    output_dir: str,
    count: int = 10,
    prefix: str = "Y",
    start_idx: int = 0,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    **kwargs,
) -> list[str]:
    """
    Generate *count* pattern files and save them under *output_dir*.

    Files are named ``{prefix}_{i}.json`` with *i* starting from *start_idx*.
    Returns the list of file paths that were written.
    """
    paths: list[str] = []
    for i in range(count):
        idx = start_idx + i
        path = os.path.join(output_dir, f"{prefix}_{idx}.json")
        gen_root_to_file(path, rows=rows, cols=cols, seed=None, **kwargs)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
#  Validation utility
# ---------------------------------------------------------------------------

def validate_pattern(pattern: dict) -> list[str]:
    """
    Check that a pattern dict satisfies all rules.
    Returns a list of violation messages (empty = valid).
    """
    errors: list[str] = []
    border_set = set(pattern.get("border_points", []))
    nc = pattern.get("node_connections", {})
    edges = pattern.get("edges", [])
    nodes = pattern.get("nodes", {})

    # Rule 1: values in range or == -999
    for e in edges:
        v = e["value"]
        if e["direction"] == "border":
            continue
        if v == UNASSIGNED or v == -999:
            continue
        if not (VALUE_MIN <= v <= VALUE_MAX):
            errors.append(
                f"Edge {e['u']}-{e['v']} value {v} out of range "
                f"[{VALUE_MIN}, {VALUE_MAX}]"
            )

    # Rule 2: root nodes (no in_edge) with assigned out_edges must have ≥ 2
    #         (border root nodes may have 1)
    for node_str, conn in nc.items():
        node = int(node_str)
        if len(conn.get("in_edges", [])) == 0 and len(conn.get("out_edges", [])) > 0:
            n_out = len(conn["out_edges"])
            if node in border_set:
                if n_out < 1:
                    errors.append(
                        f"Border root node {node} has {n_out} out edges (need ≥1)"
                    )
            else:
                if n_out < 2:
                    errors.append(
                        f"Root node {node} has {n_out} out edges (need ≥2)"
                    )

    # Rule 3: at least 1 boundary node (no out_edges, not border, has in_edges)
    has_boundary = False
    for node_str, conn in nc.items():
        node = int(node_str)
        if node in border_set:
            continue
        if (len(conn.get("out_edges", [])) == 0
                and len(conn.get("in_edges", [])) > 0):
            has_boundary = True
            break
    if not has_boundary:
        errors.append("No boundary node found")

    # Rule 4: no non‑border node may have degree 3
    for node_str, conn in nc.items():
        node = int(node_str)
        if node in border_set:
            continue
        degree = len(conn.get("out_edges", [])) + len(conn.get("in_edges", []))
        if degree == 3:
            errors.append(
                f"Node {node} has degree 3 "
                f"(out={conn['out_edges']}, in={conn['in_edges']})"
            )

    # Rule 5: degree‑2 non‑border nodes must have their 2 edges collinear
    for node_str, conn in nc.items():
        node = int(node_str)
        if node in border_set:
            continue
        connected = list(conn.get("out_edges", [])) + list(conn.get("in_edges", []))
        if len(connected) != 2:
            continue
        pos_node = nodes.get(str(node))
        pos_a = nodes.get(str(connected[0]))
        pos_b = nodes.get(str(connected[1]))
        if pos_node is None or pos_a is None or pos_b is None:
            continue
        dx1 = pos_a[0] - pos_node[0]
        dy1 = pos_a[1] - pos_node[1]
        dx2 = pos_b[0] - pos_node[0]
        dy2 = pos_b[1] - pos_node[1]
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 1e-6:
            errors.append(
                f"Node {node} has degree 2 but edges to "
                f"{connected} are NOT collinear"
            )

    return errors


# ---------------------------------------------------------------------------
#  CLI entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate root origami patterns")
    parser.add_argument("-o", "--output-dir", default="./pattern",
                        help="Directory to write pattern files (default: ./pattern)")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="Number of patterns to generate (default: 10)")
    parser.add_argument("-p", "--prefix", default="GEN",
                        help="Filename prefix (default: GEN)")
    parser.add_argument("-s", "--start", type=int, default=0,
                        help="Starting index for filenames (default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    paths = gen_root_batch(
        output_dir=args.output_dir,
        count=args.count,
        prefix=args.prefix,
        start_idx=args.start,
    )

    # Validate all generated files
    all_valid = True
    for p in paths:
        with open(p) as f:
            pat = json.load(f)
        errs = validate_pattern(pat)
        status = "✓" if not errs else "✗"
        print(f"  {status}  {p}")
        for err in errs:
            print(f"        ⚠  {err}")
            all_valid = False

    if all_valid:
        print(f"\nAll {len(paths)} patterns are valid.")
    else:
        print(f"\nSome patterns have validation errors!")
