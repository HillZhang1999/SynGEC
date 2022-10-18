from multiprocessing import Pool
import numpy as np
import pickle
import sys
import gc
from tqdm import tqdm
from fairseq.data import LabelDictionary


num_workers = 64
src_file = sys.argv[1]
conll_suffix = sys.argv[2]
mode = sys.argv[3]
structure = sys.argv[4]


input_prefix = f"{src_file}.{conll_suffix}"

if structure == "transformer":
    output_prefix = input_prefix + "_np"  # 注意
else:
    output_prefix = input_prefix + "_bart_np"  # 注意

if mode in ["dpd", "probs"]:  # 需要subword对齐的
    input_file = input_prefix + f".{mode}"
    output_file = output_prefix + f".{mode}"
else:
    input_file = input_prefix
    output_file = output_prefix

swm_list = []
if structure == "transformer":
    swm_file = src_file + ".swm"  # 注意
else:
   swm_file = src_file + ".bart_swm"  # 注意

swm_list = [[int(i) for i in line.rstrip("\n").split()] for line in open(swm_file, "r").readlines()]
label_file = "../../data/dicts/syntax_label_gec.dict"   # 注意
label_dict = LabelDictionary.load(label_file)

def create_sentence_syntax_graph_matrix(chunk, append_eos=True):
    chunk = chunk.split("\n")
    seq_len = len(chunk)
    if append_eos:
        seq_len += 1
    incoming_matrix = np.ones((seq_len, seq_len))
    incoming_matrix *= label_dict.index("<nadj>")  # outcoming矩阵可以通过转置得到
    for l in chunk:
        infos = l.rstrip().split()
        child, father = int(infos[0]) - 1, int(infos[6]) - 1  # 该弧的孩子和父亲
        if father == -1:
            father = len(chunk) # EOS代替Root
        rel = infos[7]  # 该弧的关系标签
        incoming_matrix[child,father] = label_dict.index(rel)
    return incoming_matrix


def use_swm_to_adjust_matrix(matrix, swm, append_eos=True):
    if append_eos:
        swm.append(matrix.shape[0]-1)
    new_matrix = np.zeros((len(swm), len(swm)))
    for i in range(len(swm)):
        for j in range(len(swm)):
            new_matrix[i,j] = matrix[swm[i],swm[j]]
    return new_matrix


def convert_list_to_nparray(matrix):
    return np.array(matrix)

def convert_probs_to_nparray(t):
    matrix, swm = t
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    return use_swm_to_adjust_matrix(matrix, swm)


def convert_conll_to_nparray(t):
    conll_chunk, swm = t
    incoming_matrix = create_sentence_syntax_graph_matrix(conll_chunk)
    incoming_matrix = use_swm_to_adjust_matrix(incoming_matrix, swm)
    return incoming_matrix


def data_format_convert():
    with open(output_file, 'wb') as f_out:
        res = []
        with Pool(num_workers) as pool:
            if mode == "dpd":
                with open(input_file, 'rb') as f_in:
                    gc.disable()
                    arr_list = pickle.load(f_in)
                    gc.enable()
                    assert len(swm_list) == len(arr_list), print(len(swm_list), len(arr_list))
                    for mat in pool.imap(convert_list_to_nparray, tqdm(arr_list), chunksize=256):
                        res.append(mat)
            elif mode == "probs":
                with open(input_file, 'rb') as f_in:
                    gc.disable()
                    arr_list = pickle.load(f_in)
                    gc.enable()
                    assert len(swm_list) == len(arr_list), print(len(swm_list), len(arr_list))
                    for mat in pool.imap(convert_probs_to_nparray, tqdm(zip(arr_list, swm_list)), chunksize=256):    
                        res.append(mat)
            elif mode == "conll":
                with open(input_file, 'r') as f_in:
                    conll_chunks = [conll_chunk for conll_chunk in f_in.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
                    for mat in pool.imap(convert_conll_to_nparray, tqdm(zip(conll_chunks, swm_list)), chunksize=256):
                        res.append(mat)
        pickle.dump(res, f_out)


if __name__ == "__main__":
    data_format_convert()
