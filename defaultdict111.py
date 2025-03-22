# 构造图（无向图，每条边在两个方向上都有记录）
graph = {
    'P': ['F', 'O'],
    'O': ['P', 'C'],
    'F': ['P', 'E', 'A'],
    'E': ['F', 'B', 'H'],
    'B': ['E', 'A', 'D', 'G'],
    'A': ['B', 'F', 'C'],
    'D': ['B', 'K', 'C'],
    'C': ['D', 'O', 'A'],
    'H': ['E', 'I', 'G'],
    'I': ['H', 'J', 'N'],
    'J': ['I', 'M', 'S'],
    'K': ['S', 'D', 'L'],
    'M': ['J', 'L', 'N'],
    'L': ['M', 'K'],
    'N': ['I', 'M'],
    'G': ['S', 'H', 'B'],
    'S': ['G', 'J', 'K']
}

# 为了表示每条边只用一次，我们令每条边的标记为 (u,v)（注意：无向边 (u,v) 与 (v,u) 视为同一条边）
# 我们可以用 frozenset({u,v}) 表示一条无向边。

# 记录目前找到的最长路径（边数最多）
best_path = []
best_edge_count = 0

def dfs(current, used_edges, path):
    global best_path, best_edge_count
    # 若当前路径边数比目前已知的最多的多，则更新
    if len(used_edges) > best_edge_count:
        best_edge_count = len(used_edges)
        best_path = path[:]
    # 尝试沿着每一条未用过的边走
    for nbr in graph[current]:
        edge = frozenset({current, nbr})
        if edge in used_edges:
            continue
        # 选这条边
        used_edges.add(edge)
        path.append(nbr)
        dfs(nbr, used_edges, path)
        # 回溯时撤销
        path.pop()
        used_edges.remove(edge)

# 从每个顶点出发都试一遍（因为点可重复）
for start in graph.keys():
    dfs(start, set(), [start])

print("最长路径共使用 %d 条边" % best_edge_count)
print("最长路径的顶点序列为：")
print(" -> ".join(best_path))
