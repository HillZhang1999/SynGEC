import sys

input_file = sys.argv[1]
cor_file = sys.argv[2]
id_file = sys.argv[3]
out_file = sys.argv[4]

with open(input_file, "r") as f1:
    with open(cor_file, "r") as f2:
        with open(id_file, "r") as f3:
            with open(out_file, "w") as o:
                srcs, tgts, ids = f1.readlines(), f2.readlines(), f3.readlines()
                res_li = ["" for i in range(6000)]
                for src, tgt, idx in zip(srcs, tgts, ids):
                    src = src.replace(" ", "")
                    tgt = tgt.replace(" ", "")
                    if len(src) >= 128 or len(tgt) >= 128:
                        res = src
                    else:
                        res = tgt
                    res = res.rstrip("\n")
                    res_li[int(idx)] += res
                for res in res_li:
                    o.write(res + "\n")