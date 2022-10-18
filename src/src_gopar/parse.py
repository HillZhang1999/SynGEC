from supar import Parser
import sys
import pickle
import torch
import os

def load(filename):
    sents = []
    with open(filename, 'r') as f:
        for line in f:
            res = line.rstrip().split()
            if res:
                sents.append(res)
    return sents

dep = Parser.load(sys.argv[3])
input_sentences = load(sys.argv[1])
res = dep.predict(input_sentences, verbose=False, buckets=32, batch_size=3000, prob=True)
probs = []

with open(sys.argv[2], 'w') as f:
    for r, t in zip(res, res.probs):
        f.write(str(r) + "\n")
        t1, t2 = t.split([1, len(t[0])-1], dim=-1)
        t = torch.cat((t2, t1), dim=-1)
        t = torch.cat((t, t.new_zeros((1, len(t[0])))))
        t.masked_fill_(torch.eye(len(t[0])) == 1.0, 1.0)
        t_list = t.numpy()
        probs.append(t_list)

with open(sys.argv[2] + ".probs", "wb") as o:
    pickle.dump(probs, o)