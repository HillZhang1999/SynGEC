import sys
import nltk
from multiprocessing import Pool
from tqdm import tqdm

original_file = sys.argv[1]
suffix = sys.argv[2]
in_file = original_file + "." + suffix
out_file = in_file + "_conllx"
global labels
labels = set()

def handle_special_tokens(original_tokens, conllx):
    count = 0
    pos_map = {}
    for idx, token in enumerate(conllx):
        if token.endswith(")"):
            pos_map[count] = idx
            count += 1
    for idx, token in enumerate(original_tokens):
        # if "(" in token or ")" in token:
        pos = pos_map[idx]
        # conllx[pos] = token.replace("(", "-LRB-").replace(")", "-RRB-") + conllx[pos][len(token):]
        conllx[pos] = str(idx + 1) + conllx[pos][len(token):]
    return " ".join(conllx)

def convert(line_zip):
    map = {}
    line_orig, line_con = line_zip
    line_con = line_con.replace("-LRB-", "(").replace("-RRB-", ")")
    original_tokens = line_orig.rstrip("\n").split(" ")
    conllx = line_con.rstrip("\n").split(" ")
            
    conllx_li = []
    try:
        line_con = handle_special_tokens(original_tokens, conllx)
        tree = nltk.Tree.fromstring(line_con)
    except:
        print(line_orig)
        print(line_con)
        print()

    for idx, tok in enumerate(original_tokens):
        li = ["_"] * 10
        li[0] = str(idx + 1)
        li[1] = tok
        map[str(idx + 1)] = tok
        conllx_li.append(li)

    global count
    count = len(original_tokens) + 1
    def dfs(tree):
        if isinstance(tree, nltk.Tree):
            if tree.label() != "_":  # 不考虑pre-terminal节点
                global count
                map[str(count)] = tree.label()
                tree.set_label(str(count))
                count += 1
                for child in tree:
                    dfs(child)
    dfs(tree)
    for i in range(len(original_tokens) + 1, count):
        li = ["_"] * 10
        li[0] = str(i)
        li[1] = map[str(i)]
        conllx_li.append(li)
    # tree.pretty_print()

    father_map = {}
    def build(tree):
        if isinstance(tree, nltk.Tree):
            if tree.label() != "_":  # pre-terminal节点
                for child in tree:
                    if child.label() == "_":
                        father_map[child[0]] = tree.label()
                    else:
                        father_map[child.label()] = tree.label()
                        build(child)
    build(tree)

    # print(father_map)
    for li in conllx_li:
        if li[0] not in father_map.keys():
            li[-4] = "0"
            li[-3] = "ROOT"
        else:
            li[-4] = father_map[li[0]]
            li[-3] = map[father_map[li[0]]]
    final_str = "\n".join(["\t".join(li) for li in conllx_li])
    # print(final_str + "\n")
    return final_str + "\n\n"

num_workers = 32
with open(original_file, 'r', encoding='utf-8') as f_orig:
    with open(in_file, 'r', encoding='utf-8') as f_in:
        with open(out_file, 'w', encoding='utf-8') as f_out:
            lines_orig = f_orig.readlines()
            lines_con = f_in.readlines()
            with Pool(num_workers) as pool:
                for res in pool.imap(convert, tqdm(zip(lines_orig, lines_con)), chunksize=64):
                    if res:
                        f_out.write(res)