"""
BPE后的句子和原始分词句子完成对齐操作
"""
import sys
from multiprocessing import Pool
from tqdm import tqdm

file_word = sys.argv[1]
file_bpe = sys.argv[2]
output_file = sys.argv[3]
num_workers = 16

def align(sent_tuple):
    src_line, tgt_line = sent_tuple  # src为原始分词，tgt为bpe分词
    src_line = src_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    tgt_line = tgt_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    src_tokens, tgt_tokens = src_line.rstrip().split(), tgt_line.rstrip().split()
    i, j = 0, 0
    aligned_results = []
    try:
        while j < len(tgt_tokens):
            while tgt_tokens[j].endswith("@@"):
                if src_tokens[i].endswith("@@") and tgt_tokens[j] == "@@":
                    break
                if src_tokens[i] == "@@@@@" and tgt_tokens[j] == "@@@@":
                    break
                aligned_results.append(str(i))
                j += 1
            aligned_results.append(str(i))
            i += 1
            j += 1
    except Exception:
        print(sent_tuple)
        print(src_tokens)
        print(tgt_tokens)
    assert len(aligned_results) == len(tgt_tokens)
    assert int(aligned_results[-1]) == len(src_tokens) - 1, print(src_line, src_tokens, tgt_tokens)
    return " ".join(aligned_results) + "\n"

results = []
with open(file_word, "r") as f1:
    with open(file_bpe, "r") as f2:
        src_lines, tgt_lines = f1.readlines(), f2.readlines()
        with Pool(num_workers) as pool:
            for aligned_results in pool.imap(align, tqdm(zip(src_lines, tgt_lines)), chunksize=64):
                if aligned_results:
                    results.append(aligned_results)

with open(output_file, "w") as o:
    o.writelines(results)


