from tqdm import tqdm
import sys

def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False

def is_chinese_word(word):
    for char in word:
        if not is_chinese_char(char):
            return False
    return True

def convert_word_to_char_conll(conll_chunk):
    li = conll_chunk.split("\n")
    word_li = []
    label_li = []
    head_li = []
    for line in li:
        meta_info = line.split("\t")
        word_li.append(meta_info[1])
        head_li.append(int(meta_info[6]))
        label_li.append(meta_info[7])
    char_li = []
    for word in word_li:
        if is_chinese_word:
            char_li.append([char for char in word])
        else:
            char_li.append([word])
    word_len_li = [len(word) for word in char_li]
    res_li = []
    now = 0
    for i, word in enumerate(char_li):
        for j, char in enumerate(word[:-1]):
            res_li.append((char, now + j + 2, "app"))
        dis = 0
        if head_li[i] == 0:
            res_li.append((word[-1], 0, label_li[i]))
        elif i + 1 < head_li[i]:
            for k in range(i+1, head_li[i]):  # 计算两个词的结尾字符之间相距多少个字
                dis += word_len_li[k]
            res_li.append((word[-1], now + len(word) + dis, label_li[i]))
        else:
            for k in range(head_li[i],i+1):  # 计算两个词的结尾字符之间相距多少个字
                dis += word_len_li[k]
            res_li.append((word[-1], now + len(word) - dis, label_li[i]))
        now += len(word)
    res_str = ""
    for i, res in enumerate(res_li):
        assert i+1 != res[1]
        assert 0 <= res[1] <= len(res_li)
        res_str += f"{i+1}\t{res[0]}\t_\t_\t_\t_\t{res[1]}\t{res[2]}\t_\t_\n"
    res_str += "\n"
    return res_str

input_file = sys.argv[1]
output_file = sys.argv[2]

input_chunks = [chunk for chunk in open(input_file, "r", encoding="utf-8").read().split("\n\n") if len(chunk)>0]
with open(output_file, "w", encoding="utf-8") as o:
    for chunk in tqdm(input_chunks):
        res = convert_word_to_char_conll(chunk)
        o.write(res)
    
