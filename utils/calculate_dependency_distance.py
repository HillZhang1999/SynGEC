from tqdm import tqdm
import sys
from collections import defaultdict
import pickle
from itertools import zip_longest
from multiprocessing import Pool
import numpy as np

conll_file = sys.argv[1]
swm_file = sys.argv[2]
output_file = sys.argv[3]
num_workers = 64

def use_swm_to_adjust_path(path, swm_list):
    new_path = []
    try:
        for i in swm_list:
            new_path.append(path[int(i)])
    except:
        print(swm_list) 
    return new_path


def has_root(tree):
    for node in tree:
        if node.split("\t")[6] == "0":
            return True
    return False

def construct_tree(chunk, swm_list, chunk_alter=None):
    tree = defaultdict(set)
    chunk_orig = chunk
    if not has_root(chunk):
        print(chunk)
    for line in chunk:
        li = line.split("\t")
        tree[li[6]].add(li[0])
    paths = get_path(tree)
    path_list = []
    try:
        for i in range(1, len(paths)):
            path_list.append(paths[str(i)])  # 根据规则构造的Golden树可能出现环路
        new_path = use_swm_to_adjust_path(path_list, swm_list)
    except:
        tree = defaultdict(set)
        for line in chunk_alter:
            li = line.split("\t")
            tree[li[6]].add(li[0])
        paths = get_path(tree)
        path_list = []
        for i in range(1, len(paths)):
            path_list.append(paths[str(i)])
        new_path = use_swm_to_adjust_path(path_list, swm_list)
    new_path.append(paths["0"])
    # print(tree, new_path)
    return new_path


def get_path(tree):
    paths = {}
    def dfs(node, path):
        paths[node] = path
        if node in tree.keys():
            for next_node in tree[node]:
                dfs(next_node, path + [next_node])
    dfs("0", ["0"])
    return paths


def get_nearest_ancestor(path_a, path_b):
    idx = -1
    for i in range(min(len(path_a), len(path_b))):
        if path_a[i] == path_b[i]:
            idx += 1
        else:
            break
    return path_a[idx]

def calculate_dpd(path, pruned_nodes):
    # dpd_matrix = []
    dpd_matrix = np.zeros((len(path), len(path)))
    max_dist = 0
    for i in range(len(path)):
        for j in range(len(path)):
            dpd_matrix[i][j] = len(set(path[i]) ^ set(path[j]))
    
    max_dist = np.amax(dpd_matrix)
    if pruned_nodes:    # 人为调整依存距离
        # print("Before Pruning:")
        # print(dpd_matrix)
        for i in range(len(path)):
            for j in range(len(path)):
                nodes = set(path[i]) ^ set(path[j])
                if len(nodes) == 0:
                    continue
                nodes.add(get_nearest_ancestor(path[i], path[j]))
                # print(i, path[i], j, path[j],nodes, pruned_nodes)
                if len(nodes & pruned_nodes) > 0:
                    dpd_matrix[i][j] = max_dist
        # print("After Pruning:")
    return dpd_matrix


def get_node_with_special_tag(chunk, tags):
    nodes = set()
    for line in chunk:
        li = line.split("\t")
        if li[7] in tags:
            nodes.add(li[0])
    return nodes

def solve(t):
    # chunk, swm_list, chunk_alter = t
    chunk, swm_list = t
    chunk = chunk.split("\n")
    assert chunk is not None
    # if chunk_alter:
    #     chunk_alter = chunk_alter.split("\n")
    swm_list = swm_list.strip().split()
    pruned_nodes = get_node_with_special_tag(chunk, "R")
    # pruned_nodes = None
    # return calculate_dpd(construct_tree(chunk, swm_list, chunk_alter), pruned_nodes)
    return calculate_dpd(construct_tree(chunk, swm_list), pruned_nodes)

res = []
with open(conll_file, "r") as f1:
    with open(swm_file, "r") as f2:
        chunks = [c for c in f1.read().split("\n\n") if c]
        swm_lists = f2.readlines()
        assert len(chunks) == len(swm_lists), print(len(chunks), len(swm_lists))
        with Pool(num_workers) as pool:
            for dpd in pool.imap(solve, tqdm(zip_longest(chunks, swm_lists)), chunksize=64):
                if dpd is not None:
                    res.append(dpd)

with open(output_file, "wb") as o:
    pickle.dump(res, o)


