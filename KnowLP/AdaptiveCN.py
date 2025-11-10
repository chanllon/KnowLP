import networkx as nx
import torch

def get_reference_path(start, targets, graph):

    # if start not in graph.nodes or any(target not in graph.nodes for target in targets):
    #     raise ValueError("Either the source or target node is not present in the graph.")
    if start not in graph.nodes or any(target not in graph.nodes for target in targets):
        # 如果start不在节点中，则添加一个节点关系，源节点和目标节点都是start
        graph.add_node(start)  # 添加start作为节点
        for target in targets:
            if target not in graph.nodes:
                graph.add_node(target)  # 添加target作为节点
                graph.add_edge(start, target)  # 添加源节点和目标节点之间的关系
        # raise ValueError("Either the source or target node is not present in the graph.")

    all_paths = []
    for target in targets:
        if nx.has_path(graph, start, target):
            paths = nx.all_shortest_paths(graph, start, target)
            all_paths.extend(paths)

    reference_path = None
    reference_path_length = float('inf')
    for path in all_paths:
        if set(targets).issubset(path):
            # path_length = nx.shortest_path_length(graph, source=start, target=path[-1]) # Dijkstra algorithm
            path_length = nx.astar_path_length(graph, source=start, target=path[-1]) # A* algorithm
            if path_length < reference_path_length:
                reference_path = path
                reference_path_length = path_length

        else:
            reference_path = []
            for target in targets:
                reference_path.extend(nx.shortest_path(graph, start, target))

            reference_path = list(set(reference_path))
    if reference_path == None:
        reference_path = [start]

    reference_path_stack = reference_path[::-1]

    return reference_path_stack



def get_predecessors_within_k_hop(node, graph, k):
    predecessors = set()
    queue = [(node, 0)]

    while queue:
        current_node, hop = queue.pop(0)
        if hop > k-1:
            break
        if current_node not in graph.nodes:
            continue
        else:
            for predecessor in graph.predecessors(int(current_node)):
                predecessors.add(predecessor)
                queue.append((predecessor, hop + 1))

    return predecessors


def get_successors_within_1_hop(node, graph):
    successors = set()
    if node in graph:
        successors = graph[node]

    return successors




def Adaptive_Cognitive_Nevigation(target_nodes, knowledge_structure, learning_item, mastery, k_hop):


    reference_path = get_reference_path(learning_item, target_nodes, knowledge_structure.to_undirected())

    if learning_item == reference_path[-1]: # following reference
        if mastery:
            if len(reference_path)==1:
                return [reference_path[-1]]  # learning next
            else:
                reference_path.pop()  # mastered
                return [reference_path[-1]]  # learning next

        else:
            candidates = get_predecessors_within_k_hop(learning_item,knowledge_structure,k_hop)
            candidates = list(candidates)
            if not candidates:
                candidates.append(learning_item)
            return list(candidates)
    else:

        return [reference_path[-1]]


def Find_Longest_Path_With_Threshold(graph, target_nodes, knowledge_states, th):
    """
    从目标节点出发，找到最长路径的起点，并满足阈值条件。
    :param graph: 有向图 nx.DiGraph
    :param target_nodes: 目标节点的列表
    :param knowledge_states: 知识状态（字典）：{节点: 状态值}
    :param th: 阈值，当前驱节点知识状态均大于 th 时更新起始节点
    :return: 起点和路径
    """
    # 转换知识状态为字典形式 {节点: 知识值}
    knowledge_states_dict = {i: knowledge_states[0, i].item() for i in range(knowledge_states.size(1))}

    def dfs(node, visited, path):
        """
        深度优先搜索，用于递归查找路径。
        :param node: 当前节点
        :param visited: 已访问节点集合
        :param path: 当前路径
        :param contains_targets: 是否包含其他目标节点
        :return: 最长路径和是否满足条件
        """
        if node in visited:
            return path

        # 知识状态超过阈值，停止向前扩展
        if knowledge_states_dict.get(node, 0) > th:
            return path

        #设置往前倒推的最大长度为30
        if len(path) >= 30:
            return path

        visited.add(node)
        path.append(node)
        # if node in target_nodes and node != start_target:  # 避免重复计入起始目标节点
        #     contains_targets = True

        # # 如果当前节点的所有前驱节点知识状态 > 阈值，则更新 start_node
        # all_predecessors = list(graph.predecessors(node))
        # if all_predecessors and all(knowledge_states.get(pred, 0) > th for pred in all_predecessors):
        #     # 如果满足阈值条件，将当前节点直接作为新起点
        #     return path, contains_targets

        # 尝试扩展所有前驱节点
        max_path = path[:]
        if node not in graph.nodes:
            return max_path

        for predecessor in graph.predecessors(node):
            new_path = dfs(predecessor, visited.copy(), path[:])
            if len(new_path) > len(max_path):
                max_path = new_path

        return max_path

    # 遍历每个目标节点，找到从中倒推的最长路径
    longest_path = []
    for target_node in target_nodes:
        path = dfs(target_node, set(), [])
        if len(path) > len(longest_path):
            longest_path = path

    # 最长路径的起始节点
    start_node = longest_path[-1] if longest_path else None
    if start_node is None:
        start_node = target_nodes[0]/2
    return start_node, list(reversed(longest_path))


def Find_Threshold_Neighbors(G, target, knowledge_states, th):
    """
    找到目标节点 target 所有知识状态小于阈值 th 的前一个和后一个节点，确保节点不重复。

    参数:
    - G: networkx.DiGraph，有向图
    - target: 目标节点
    - knowledge_states: 字典，{node: 状态值}
    - th: 阈值

    返回:
    - list，符合条件的所有不重复节点
    """
    # 转换知识状态为字典形式 {节点: 知识值}
    knowledge_states = {i: knowledge_states[0, i].item() for i in range(knowledge_states.size(1))}

    all_below_th = set()  # 使用集合来存储不重复的节点

    # 1. 找到所有前一个节点（predecessors），判断状态值是否小于阈值
    if target not in G.nodes:
        if knowledge_states.get(target, 0) < th:  # 获取状态值，默认0
            all_below_th.add(target)
    else:
        for pred in G.predecessors(target):
            if knowledge_states.get(pred, 0) < th:  # 获取状态值，默认0
                all_below_th.add(pred)  # 添加到集合中

        # 2. 找到所有后一个节点（successors），判断状态值是否小于阈值
        for succ in G.successors(target):
            if knowledge_states.get(succ, 0) < th:  # 获取状态值，默认0
                all_below_th.add(succ)  # 添加到集合中
    if all_below_th == set():  # 如果集合为空，则返回目标节点
        all_below_th.add(target)
    return list(all_below_th)


if __name__ == '__main__':

    # Demo
    G = nx.DiGraph()
    G.add_edges_from([(0,1),(0,2),(1,3),(2,4),(3,4),(2,8),(4,8),(5,4),(5,9),(6,7),(7,8),(8,9)])

    start_node = 4
    target_nodes = [6,9]

    #随机生成知识一个学生的知识状态
    # 生成值在 0 到 1 之间的随机 tensor
    knowledge_states_tensor = torch.rand(1, 10)

    threshold = 0.7
    start_node, longest_path = find_longest_path_with_threshold(G, target_nodes, knowledge_states_tensor, threshold)

    reference_path = get_reference_path(start_node,target_nodes,G.to_undirected())

    print(reference_path)

    candidates = Adaptive_Cognitive_Nevigation(target_nodes, G,start_node,0,1)

    print(candidates)
    # print(sp)