#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from multiprocessing import Pool
from tqdm import tqdm
import torch
from fairseq import options, tasks, utils
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset, IndexedRawLabelDataset
import pickle
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")
label_dict = None

def main(args):
    utils.import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir, "preprocess.log"),
        )
    )
    logger.info(args)

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    target = not args.only_source

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))
    
    if args.labeldict:
        # print(task)
        global label_dict
        label_dict = [task.load_syntax_label_dictionary(d) for d in args.labeldict]

    if args.joined_dictionary:
        assert (
            not args.srcdict or not args.tgtdict
        ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]},
                src=True,
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))
    if label_dict is not None:
        for i, d in enumerate(label_dict):
            d.save(dict_path(f"label{i}"))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, already_numberized=False, append_eos=True):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        already_numberized,
                        append_eos,
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin"),
            impl=args.dataset_impl,
            vocab_size=len(vocab),
        )
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t), offset=0, end=offsets[1], already_numberized=already_numberized, append_eos=append_eos
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result["nseq"]

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, None, "bin"), impl=args.dataset_impl
        )

        merge_result(
            Binarizer.binarize_alignments(
                input_file,
                utils.parse_alignment,
                lambda t: ds.add_item(t),
                offset=0,
                end=offsets[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info("[alignments] {}: parsed {} alignments".format(input_file, nseq[0]))


    def make_binary_matrix_dataset(input_file, output_prefix, num_workers, lang, info_format):
        nseq = [0]
        if "conll" in info_format:  # 句法掩码矩阵
            dtype = np.int64  # Pytorch1.7.1 要求embedding层的输入必须是Long，这会导致不必要的内存开销，因为我们的句法词典的范围是很小的，后续看看能否hack一下。
        elif "probs" in info_format:  # 句法概率掩码矩阵
            dtype = np.float16
        elif "dpd" in info_format:  # 句法距离掩码矩阵
            dtype = np.float16
        else:
            raise NotImplementedError

        def merge_result(worker_result):
            nseq[0] += worker_result["nseq"]

        input_list = pickle.load(open(input_file, "rb"))  # input_matrix_list是预先处理好的包含矩阵的list文件，
        offsets = Binarizer.find_offsets_list(input_list, num_workers)  # 并行化会有问题
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_matrix,
                    (
                        args,
                        input_list[offsets[worker_id]:offsets[worker_id + 1]],
                        prefix,
                        dtype,
                        lang
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin"), impl=args.dataset_impl, dtype=dtype
        )
        ds.dtype = dtype

        merge_result(
            Binarizer.binarize_matrix(
                input_list[:offsets[1]],
                lambda t: ds.add_item(t)
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info("[{}] {}: parsed {} sentences".format(info_format, input_file, nseq[0]))

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            if lang == "src_nt_bart":
                make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, already_numberized=True, append_eos=False)
            else:
                make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(
                    vocab, validpref, outprefix, lang, num_workers=args.workers
                )
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.trainpref + "." + args.align_suffix,
                "train.align",
                num_workers=args.workers,
            )
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.validpref + "." + args.align_suffix,
                "valid.align",
                num_workers=args.workers,
            )
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.testpref + "." + args.align_suffix,
                "test.align",
                num_workers=args.workers,
            )

    def make_all_matrix(info_format):  # 二进制化所有使用numpy矩阵存储的数据，主要是句法相关的掩码矩阵、距离矩阵和概率矩阵
        if info_format == "conll":
            suffixs = args.conll_suffix
        elif info_format == "probs":
            suffixs = args.probs_suffix
        elif info_format == "dpd":
            suffixs = args.dpd_suffix
        else:
            raise NotImplementedError
        if args.trainpref:
            for suffix in suffixs:
                train_src_file = args.trainpref.replace("bpe", suffix) + ".src"
                if os.path.exists(train_src_file):
                    output_prefix = f"train.{suffix}"
                    make_binary_matrix_dataset(
                        train_src_file,
                        output_prefix,
                        num_workers=args.workers,
                        info_format=info_format,
                        lang=args.source_lang
                    )
                else:
                    raise FileExistsError
        if args.validpref:
            for suffix in suffixs:
                valid_src_file = args.validpref.replace("bpe", suffix) + ".src"
                if os.path.exists(valid_src_file):
                    output_prefix = f"valid.{suffix}"
                    make_binary_matrix_dataset(
                        valid_src_file,
                        output_prefix,
                        num_workers=args.workers,
                        info_format=info_format,
                        lang=args.source_lang
                    )
                else:
                    raise FileExistsError
        if args.testpref:
            for suffix in suffixs:
                test_src_file = args.testpref.replace("bpe", suffix) + ".src"
                if os.path.exists(test_src_file):
                    output_prefix = f"test.{suffix}"
                    make_binary_matrix_dataset(
                        test_src_file,
                        output_prefix,
                        num_workers=args.workers,
                        info_format=info_format,
                        lang=args.source_lang
                    )
                else:
                    raise FileExistsError

    make_all(args.source_lang, src_dict)
    if args.source_lang_with_nt is not None:  # 读取含有non-terminal节点的句子
        make_all(args.source_lang_with_nt, src_dict)
    if target:
        make_all(args.target_lang, tgt_dict)
    if args.align_suffix:
        make_all_alignments()
    if args.conll_suffix:
        make_all_matrix("conll")
    if args.dpd_suffix:
        make_all_matrix("dpd")
    if args.probs_suffix:
        make_all_matrix("probs")
    logger.info("Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding="utf-8") as align_file:
            with open(src_file_name, "r", encoding="utf-8") as src_file:
                with open(tgt_file_name, "r", encoding="utf-8") as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
            os.path.join(
                args.destdir,
                "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
            ),
            "w",
            encoding="utf-8",
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, already_numberized=False, append_eos=True):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end, already_numberized=already_numberized
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(
        filename, parse_alignment, consumer, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res

def binarize_matrix(args, input_list, output_prefix, dtype, lang):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
        dtype=dtype
    )
    ds.dtype = dtype

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_matrix(
        input_list, consumer
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)

def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
