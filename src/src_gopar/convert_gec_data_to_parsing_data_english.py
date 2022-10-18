import sys
import copy
from multiprocessing import Pool
from tqdm import tqdm

num_workers = 32

class Tree_Transformer:
    """根据m2格式信息完成句法树变形，需要提前提供Golden句法树

    Arguments:
        Errorifier {Class} -- 父类，仅做加噪
    """
    
    def __init__(self, sentence_conllx: list, m2: list):
        """构造函数

        Arguments:
            sentence_conllx {list} -- CoNLLx格式的句子
        """
        self.original_sentence = " ".join([t[1] for t in sentence_conllx])
        assert self.original_sentence == m2[0][2:], print(self.original_sentence, m2[0][2:])
        self.edits = m2[1:]
        self.original_conllx = copy.deepcopy(sentence_conllx)
        self.sentence = self.original_sentence
        self.conllx = sentence_conllx
        self.tokenized = None
        self.tokenize()
        self.parse_conllx()

    def tokenize(self):
        self.tokenized = self.sentence.split()

    def correct_conllx(self):
        return "\n".join(["\t".join(t) for t in self.original_conllx])

    def error_conllx(self):
        conllx = copy.deepcopy(self.conllx)
        error_conllx = list(map(self.convert_int_to_str_in_conllx, conllx))
        return "\n".join(["\t".join(t) for t in error_conllx])

    @staticmethod
    def convert_str_to_int_in_conllx(conllx):
        conllx[0] = int(conllx[0])
        conllx[6] = int(conllx[6])
        return conllx

    @staticmethod
    def convert_int_to_str_in_conllx(conllx):
        conllx[0] = str(conllx[0])
        conllx[6] = str(conllx[6])
        return conllx

    def parse_conllx(self):
        self.conllx = list(map(self.convert_str_to_int_in_conllx, self.conllx))

    def get_children(self, node):
        children = []
        for i, t in enumerate(self.conllx):
            if t[6] == node:
                children.append(i)
        return children

    def transform_tree(self, error_type, error_pos, target_token=""):
        assert error_type in ["D", "S", "I", "W"], print("Wrong Error Type!")
        if error_type == "D":  # 插入了词语
            for t in self.conllx:
                if t[0] > error_pos: t[0] += 1
                if t[6] > error_pos: t[6] += 1
            new_line = [error_pos + 1, target_token, "_", "_", "_", "_", error_pos + 2, "R",  "_", "_"]  # "R"代表冗余错误
            self.conllx.insert(error_pos, new_line)
        elif error_type == "S":  # 替换了词语
            self.conllx[error_pos][1] = target_token
            self.conllx[error_pos][7] = "S"  # 替换该节点的入弧标签为”S“
        elif error_type == "I":  # 删除了词语
            children = self.get_children(error_pos+1) 
            father = self.conllx[error_pos][6]
            for c in children:  # 被删除节点的所有孩子，指向它的父亲节点
                self.conllx[c][6] = father
            if error_pos < len(self.conllx) - 1:
                self.conllx[error_pos + 1][7] = "M"  # 替换该节点的右侧节点入弧标签为”M“，代表缺失了左侧缺失了token
            for t in self.conllx:
                if t[0] > error_pos: t[0] -= 1
                if t[6] > error_pos: t[6] -= 1
            del self.conllx[error_pos]
    
    def convert(self):
        offset = 0
        for edit in self.edits:
            if not edit[0] == "A":
                continue
            meta_info = edit.split("|||")
            start_pos, end_pos = meta_info[0].lstrip("A ").split()
            start_pos, end_pos = int(start_pos), int(end_pos)
            error_type = meta_info[1]
            target_words = meta_info[2].split(" ")
            target_word_num = len(target_words)
            if error_type[0] == "R":
                if end_pos - start_pos == target_word_num:  # 目前只考虑等长替换，不等长替换不好写规则
                    target_word_id = 0
                    for i in range(start_pos, end_pos):
                        pos_id = i + offset
                        self.transform_tree("S", pos_id, target_words[target_word_id])
                        target_word_id += 1
                if end_pos - start_pos > target_word_num:  # 正确Span比错误Span长，先替换，再插入
                    target_word_id = 0
                    for i in range(start_pos, start_pos + target_word_num):
                        pos_id = i + offset
                        self.transform_tree("S", pos_id, target_words[target_word_id])
                        target_word_id += 1
                    for i in range(start_pos + target_word_num, end_pos):
                        pos_id = i + offset
                        self.transform_tree("I", pos_id)
                        offset -= 1
                else:  # 正确Span比错误Span短，先替换，再删除
                    target_word_id = 0
                    for i in range(start_pos, end_pos):
                        pos_id = i + offset
                        self.transform_tree("S", pos_id, target_words[target_word_id])
                        target_word_id += 1
                    for i in range(target_word_id, target_word_num):
                        insert_word = target_words[i]
                        pos_id = end_pos + offset
                        if pos_id + 2 > len(self.conllx):
                            continue
                        self.transform_tree("D", pos_id, insert_word)
                        offset += 1
            elif error_type[0] == "M":  # 冗余错误转化
                for insert_word in target_words:
                    pos_id = start_pos + offset
                    if pos_id + 2 > len(self.conllx):
                        continue
                    self.transform_tree("D", pos_id, insert_word)
                    offset += 1
            elif error_type[0] == "U":  # 缺失错误转化
                for i in range(start_pos, end_pos):
                    pos_id = i + offset
                    self.transform_tree("I", pos_id)
                    offset -= 1


def solve(tup):
    conll_chunk, m2_chunk = tup
    tree_trans = Tree_Transformer(conll_chunk, m2_chunk)
    tree_trans.convert()
    return tree_trans.error_conllx()

if __name__ == "__main__":
    conll_file = sys.argv[1]
    m2_file = sys.argv[2]
    out_file = sys.argv[3]

    with open(conll_file, "r") as f1:
        with open(m2_file, "r") as f2:
           with open(out_file, "w") as f3: 
               conll_chunks = [[s.split("\t") for s in conll_chunk.split("\n") if s] for conll_chunk in f1.read().split("\n\n") if conll_chunk and conll_chunk != "\n"]
               m2_chunks = [[s for s in m2_chunk.split("\n") if s] for m2_chunk in f2.read().split("\n\n") if m2_chunk and m2_chunk != "\n"]
               assert len(conll_chunks) == len(m2_chunks), print(len(conll_chunks), len(m2_chunks))
               with Pool(num_workers) as pool:
                    for res in pool.imap(solve, tqdm(zip(conll_chunks, m2_chunks)), chunksize=64):
                        if res:
                            f3.write(res + "\n\n")