import sys
import os

di = sys.argv[1]
ref_conll_file = sys.argv[2]
out_file = sys.argv[3]

root = [root for root, dirs, files in os.walk(di)][0]
files_all = [files for root, dirs, files in os.walk(di)][0]
files_num = len(files_all)
files_all.sort()
n_best = files_num // 4
lines_and_dis = []

for i in range(n_best):
    cor_conll_file = root + "/" + files_all[i * 4 + 1]
    tgt_file = root + "/" + files_all[i * 4]
    tgt_lines = open(tgt_file, "r", encoding="utf-8").readlines()
    distances = []
    with open(cor_conll_file, 'r', encoding='utf-8') as f1:
        with open(ref_conll_file, 'r', encoding='utf-8') as f2:
            cor_conll_chunks = [[s.split("\t") for s in conll_chunk.split("\n") if s] for conll_chunk in f1.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
            ref_conll_chunks = [[s.split("\t") for s in conll_chunk.split("\n") if s] for conll_chunk in f2.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
            if not (len(cor_conll_chunks) == len(ref_conll_chunks) == len(tgt_lines)):
                print(cor_conll_file, len(cor_conll_chunks), len(ref_conll_chunks), len(tgt_lines))
                continue
            offset = 0  # 纠正结果为空行
            for i in range(len(tgt_lines)):
                tgt_line = tgt_lines[i].rstrip("\n")
                if tgt_line != "":
                    cor_conll_chunk = cor_conll_chunks[i - offset]
                    ref_conll_chunk = ref_conll_chunks[i]
                    dis = 0
                    for cor_conll_line, ref_conll_line in zip(cor_conll_chunk, ref_conll_chunk):
                        if (cor_conll_line[-3] == ref_conll_line[-3]) or (cor_conll_line[-3] == "O" and ref_conll_line[-3] not in ["M", "S", "R"]):
                            continue
                        dis += 1
                    distances.append(dis)
                else:
                    ref_conll_chunk = ref_conll_chunks[i]
                    dis = 0
                    for ref_conll_line in ref_conll_chunk:
                        if ref_conll_line == "R":
                            continue
                        dis += 1
                    distances.append(dis)
                    offset += 1
    lines_and_dis.append(list(zip(tgt_lines, distances)))

with open(out_file, "w", encoding="utf-8") as out_file:
    for i in range(len(lines_and_dis[0])):
        min_dis, n = lines_and_dis[0][i][1], 0
        for j in range(1, len(lines_and_dis)):
            now_dis = lines_and_dis[j][i][1]
            if now_dis < min_dis:
                n = j
                min_dis = now_dis
        out_file.write(lines_and_dis[n][i][0])