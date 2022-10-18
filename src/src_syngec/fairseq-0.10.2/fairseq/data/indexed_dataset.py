# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import struct
from functools import lru_cache
import gc
import numpy as np
import torch
from fairseq.data.fasta_dataset import FastaDataset
from fairseq.file_io import PathManager
import pickle
from multiprocessing import Pool
from . import FairseqDataset
from tqdm import tqdm


def from_dict_to_mask_for_syntax(conll):
    """将句法信息转换成Mask矩阵的形式
    Yue Zhang
    2021.12.28
    Arguments:
        conll {[type]} -- [description]
    """
    seq_len = len(conll) + 1
    now_incoming_arc_mask = torch.zeros((seq_len, seq_len), dtype=torch.int8)
    now_outcoming_arc_mask = torch.zeros((seq_len, seq_len), dtype=torch.int8)
    for token_id, token_meta in enumerate(conll):
        for arc in token_meta["father"]:
            now_incoming_arc_mask[token_id, arc[0]] = arc[1]
        for arc in token_meta["children"]:
            now_outcoming_arc_mask[token_id, arc[0]] = arc[1]
    return (now_outcoming_arc_mask, now_incoming_arc_mask)

def __best_fitting_dtype(vocab_size=None, dtype=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        if dtype is None:
            return np.int32
        else:
            return dtype


def get_available_dataset_impl():
    return ["raw", "lazy", "cached", "mmap", "fasta"]


def infer_dataset_impl(path):
    if IndexedRawTextDataset.exists(path):
        return "raw"
    elif IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
            else:
                return None
    elif FastaDataset.exists(path):
        return "fasta"
    else:
        return None


def make_builder(out_file, impl, vocab_size=None, dtype=np.int32):
    if impl == "mmap":
        # return MMapIndexedDatasetBuilder(
        #     out_file, dtype=__best_fitting_dtype(vocab_size)
        # )
        return MMapIndexedDatasetBuilder(
            out_file, dtype=__best_fitting_dtype(vocab_size, dtype)
        )
    elif impl == "fasta":
        raise NotImplementedError
    else:
        return IndexedDatasetBuilder(out_file, dtype)


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None):
    if impl == "raw" and IndexedRawTextDataset.exists(path):
        assert dictionary is not None
        return IndexedRawTextDataset(path, dictionary)
    elif impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == "cached" and IndexedDataset.exists(path):
        return IndexedCachedDataset(path, fix_lua_indexing=fix_lua_indexing)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path)
    elif impl == "fasta" and FastaDataset.exists(path):
        from fairseq.data.fasta_dataset import EncodedFastaDataset

        return EncodedFastaDataset(path, dictionary)
    return None


def dataset_exists(path, impl):
    if impl == "raw":
        return IndexedRawTextDataset.exists(path)
    elif impl == "mmap":
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16,
    9: np.float16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


class IndexedDataset(FairseqDataset):
    """Loader for TorchNet IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path, fix_lua_indexing=False):
        super().__init__()
        self.path = path
        self.fix_lua_indexing = fix_lua_indexing
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. "
                "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), "rb", buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        item = torch.from_numpy(a)
        if self.dtype == np.int32:  # 默认的indexed数据集最后都会将tensor转long，因为fairseq只考虑到了token_id，但我们的概率矩阵等需要用float存储，因此要修改下
            item = item.long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):
    def __init__(self, path, fix_lua_indexing=False):
        super().__init__(path, fix_lua_indexing=fix_lua_indexing)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx : ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx : ptx + a.size])
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item


class IndexedRawTextDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.lines.append(line.strip("\n"))
                tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    append_eos=self.append_eos,
                    reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


"""修改记录
Yue Zhang
2021.12.25
添加一个从文本文件中读取标签的类
主要用于：1）读取alignfile；2）读取subwordmapfile
"""
class IndexedRawLabelDataset(torch.utils.data.Dataset):
    def __init__(self, path, append_eos=False):
        self.append_eos=append_eos
        self.labels_list = self.read_data(path)
        self.size = len(self.labels_list)

    def read_data(self, path):
        lines = open(path, 'r').readlines()
        labels_list = [[int(l) for l in line.split()] for line in lines]
        if self.append_eos:
            [l.append(0) for l in labels_list]  # Subword Map里需要考虑下Eos要怎么对齐
        # tensor_list = [torch.IntTensor(l) for l in labels_list]
        return labels_list

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
    
    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return torch.IntTensor(self.labels_list[i])

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)

"""修改记录
Yue Zhang
2021.12.26
添加一个从CoNLL格式二进制文件中读取Father信息和Children信息的类
* 内存爆炸问题:
直接读取所有掩码矩阵信息，会导致内存不足。基本定位到问题的原因是当前的这个数据集类一次性通过pickle.load载入的信息太多。
* 修改方案:
preprocess端，序列化的文件只包含生成掩码矩阵所需要的信息，
真正三维化的掩码矩阵，需要在getitem函数中实现
* 修改结果：
初步修改后，10w规模的数据不会出现内存不足问题了。
但是，全量数据(70w)依然会出现内存不足，估计原因和conll_list的存储方式有关系。
"""
class IndexedCoNLLDataset(torch.utils.data.Dataset):
    def __init__(self, conll_path, dpd_path=None, probs_path=None):
        # self.conll_list, self.incoming_arc_mask_list, self.outcoming_arc_mask_list = self.read_data(path)
        self.conll_list = self.read_data(conll_path)
        self.arc_mask_preprocessed = False
        if len(self.conll_list) == 3:
            self.incoming_arc_mask_list = self.conll_list[1]
            self.outcoming_arc_mask_list = self.conll_list[2]
            self.conll_list = self.conll_list[0]
            self.arc_mask_preprocessed = True
        self.dpd_list = None
        if dpd_path is not None and dpd_path != "":
            self.dpd_list = self.read_data(dpd_path)
            assert len(self.conll_list) == len(self.dpd_list)
        self.probs_list = None
        if probs_path is not None and probs_path != "":
            self.probs_list = self.read_data(probs_path)
            assert len(self.probs_list) == len(self.dpd_list)
        self.size = len(self.conll_list)

    def read_data(self, path):
        data = pickle.load(open(path, "rb"))
        return data

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')
    
    @lru_cache(maxsize=8)
    def get_outcoming_arc_mask(self, i):
        self.check_index(i)
        # return self.outcoming_arc_mask_list[i]
        if self.arc_mask_preprocessed:
            return torch.LongTensor(self.outcoming_arc_mask_list[i])
        return self.from_dict_to_mask_for_syntax(self.conll_list[i], "children")  # 待优化，其实可以先在预处理阶段把这个矩阵算好，用list或者numpy形式存储，在getitem里直接转成tensor即可。
    
    @lru_cache(maxsize=8)
    def get_incoming_arc_mask(self, i):
        self.check_index(i)
        # return self.incoming_arc_mask_list[i]
        if self.arc_mask_preprocessed:
            return torch.LongTensor(self.incoming_arc_mask_list[i])
        return self.from_dict_to_mask_for_syntax(self.conll_list[i], "father") 

    @lru_cache(maxsize=8)
    def get_dpd_matrix(self, i):
        self.check_index(i)
        # return torch.FloatTensor(self.dpd_list[i])
        return torch.HalfTensor(self.dpd_list[i])

    @lru_cache(maxsize=8)
    def get_probs_matrix(self, i):
        self.check_index(i)
        # return torch.FloatTensor(self.dpd_list[i])
        return torch.HalfTensor(self.probs_list[i])
    
    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.conll_list[i]

    @staticmethod
    def from_dict_to_mask_for_syntax(conll, rel):
        """将句法信息转换成Mask矩阵的形式
        Yue Zhang
        2021.12.28
        Arguments:
            conll {[type]} -- [description]
        """
        seq_len = len(conll)
        arc_mask = torch.zeros((seq_len, seq_len), dtype=torch.long)
        # arc_mask = torch.ones((seq_len, seq_len), dtype=torch.int8)
        arc_mask *= 2  # 未邻接的顶点，用<nadj>填充，和<pad>区分开
        for token_id, token_meta in enumerate(conll):
            for arc in token_meta[rel]:
                arc_mask[token_id, arc[0]] = arc[1]
        return arc_mask

    def __setitem__(self, i, v):
        self.check_index(i)
        self.conll_list[i] = v  # 重新设置CoNLL信息后，相应的掩码矩阵必须要重置 Todo：强制同步

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)

class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.float16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8,
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, "wb")
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]

    def add_item(self, tensor):
        # +1 for Lua compatibility
        bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        with open(data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, "wb")
        index.write(b"TNTIDX\x00\x00")
        index.write(struct.pack("<Q", 1))
        index.write(struct.pack("<QQ", code(self.dtype), self.element_size))
        index.write(struct.pack("<QQ", len(self.data_offsets) - 1, len(self.sizes)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        index.close()


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype in [np.int64, np.int32, np.uint16]:
            np_array = np_array.astype(np.int64)
        else:
            np_array = np_array.astype(np.float16)
        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )


def get_indexed_dataset_to_local(path):
    local_index_path = PathManager.get_local_path(index_file_path(path))
    local_data_path = PathManager.get_local_path(data_file_path(path))

    assert local_index_path.endswith(".idx") and local_data_path.endswith(".bin"), (
        "PathManager.get_local_path does not return files with expected patterns: "
        f"{local_index_path} and {local_data_path}"
    )

    local_path = local_data_path[:-4]  # stripping surfix ".bin"
    assert local_path == local_index_path[:-4]  # stripping surfix ".idx"
    return local_path


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
