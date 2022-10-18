import sys

input_file = sys.argv[1]
cor_file = sys.argv[2]
out_file = sys.argv[3]

with open(input_file, "r") as f1:
    with open(cor_file, "r") as f2:
        with open(out_file, "w") as o:
            srcs, tgts = f1.readlines(), f2.readlines()
            for src, tgt in zip(srcs, tgts):
                if len(src.split(" ")) >= 64 or len(tgt.split(" ")) >= 64:
                    res = src
                else:
                    res = tgt
                res = res.rstrip("\n")
                o.write(res + "\n")