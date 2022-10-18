# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    SyntaxEnhancedLanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LabelDictionary,
    IndexedRawLabelDataset,
    IndexedCoNLLDataset,
    Dictionary,
    DictionaryFromTransformers
)
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks import register_task
from collections import defaultdict

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


"""修改记录
Yue Zhang
2021.12.26
读取含有句法的句对数据集
"""
def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    syntax_label_dict,  # 句法标签词典
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    load_syntax=True,
    load_probs=True,
    load_subword_align=True,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    args=None
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    src_with_nt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        src_nt_dataset = data_utils.load_indexed_dataset(
            prefix + args.source_lang_with_nt, src_dict, dataset_impl
        )
        if src_nt_dataset is not None:
            src_with_nt_datasets.append(src_nt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        src_nt_dataset = src_with_nt_datasets[0] if len(src_with_nt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    """修改记录
    Yue Zhang
    2021.12.26
    读取两个用于句法增强的数据集（目前只关注源端句法）
    Todo: 支持combine
    Todo: 支持目标端句法
    """
    src_conll_dataset = None
    src_dpd_dataset = None
    src_probs_dataset = None
    syntax_type = None
    if load_syntax:
        syntax_type = args.syntax_type
        src_conll_dataset = []
        src_dpd_dataset = []
        src_probs_dataset = []
        # 句法掩码矩阵(GAT&GCN)
        for conll_suffix in args.conll_suffix:
            src_conll_path = os.path.join(data_path, "{}.{}.{}-{}.src".format(split, conll_suffix.strip(), src, tgt))
            if indexed_dataset.dataset_exists(src_conll_path, impl=dataset_impl):
                src_conll_dataset.append(data_utils.load_indexed_dataset(
                    src_conll_path, None, dataset_impl
                ))
            else:
                print(src_conll_path)
                raise FileNotFoundError
        # 依存距离矩阵(DSA)
        for dpd_suffix in args.dpd_suffix:
            src_dpd_path = os.path.join(data_path, "{}.{}.{}-{}.src".format(split, dpd_suffix.strip(), src, tgt))
            if indexed_dataset.dataset_exists(src_dpd_path, impl=dataset_impl):
                src_dpd_dataset.append(data_utils.load_indexed_dataset(
                    src_dpd_path, None, dataset_impl
                ))
            else:
                print(src_dpd_path)
                raise FileNotFoundError
        # 句法概率矩阵(Soft GCN)
        for probs_suffix in args.probs_suffix:
            src_probs_path = os.path.join(data_path, "{}.{}.{}-{}.src".format(split, probs_suffix.strip(), src, tgt))
            if indexed_dataset.dataset_exists(src_probs_path, impl=dataset_impl):
                src_probs_dataset.append(data_utils.load_indexed_dataset(
                    src_probs_path, None, dataset_impl
                ))
            else:
                print(src_probs_path)
                raise FileNotFoundError

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    src_nt_dataset_sizes = src_nt_dataset.sizes if src_nt_dataset is not None else None
    return SyntaxEnhancedLanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        src_nt = src_nt_dataset,
        src_nt_sizes = src_nt_dataset_sizes,
        src_conll_dataset = src_conll_dataset,
        src_dpd_dataset = src_dpd_dataset,
        src_probs_dataset = src_probs_dataset,
        syntax_label_dict=syntax_label_dict,  # 句法标签词典
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        syntax_type=syntax_type
    )


@register_task("syntax-enhanced-translation")
class SyntaxEnhancedTranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--syntax-model-file', type=str, default=None)
        parser.add_argument("--bart-model-file-from-transformers", type=str, default=None)
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--use-syntax', action='store_true',
                            help='use the syntactic information')
        parser.add_argument('--syntax-encoder', default="GCN", choices=["GAT", "GCN"], type=str,
                            help='use which encoder to encode syntactic information')
        parser.add_argument('--use-dpd', action='store_true',
                            help='use the dependency distance information')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument("--conll-suffix", metavar="FP", default=["conll"], nargs='+',
                        help="conll file suffix")
        parser.add_argument("--dpd-suffix", metavar="FP", default=["dpd"], nargs='+',
                        help="dependency distance file suffix")
        parser.add_argument("--probs-suffix", metavar="FP", default=["probs"], nargs='+',
                        help="dependency probabilities file suffix")
        parser.add_argument("--swm-suffix", metavar="FP", default="swm",
                        help="subword map file suffix")
        parser.add_argument("--syntax-type", default=["dep"], nargs='+',
                        help="dependency syntax or consitituency syntax")
        parser.add_argument("--source-lang-with-nt", default="src_nt", metavar="SRC",
                    help="source language")
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, syntax_label_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.syntax_label_dict = syntax_label_dict  # 句法标签词典

    @classmethod
    def load_dictionary(cls, filename, is_from_transfromers=False):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        # if is_from_transfromers:
        #     return DictionaryFromTransformers.load(filename)  # 通过transformers转换而来时的模型，词表中的特殊token需要调整为transformers格式
        return Dictionary.load(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        def clean_suffixs(suffixs):
            res = []
            for suffix in suffixs:
                if suffix.strip():
                    res.append(suffix.strip())
            return res
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )
        
        if args.bart_model_file_from_transformers is not None:
            is_from_transfromers = True
        else:
            is_from_transfromers = False
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.source_lang)), is_from_transfromers
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(args.target_lang)), is_from_transfromers
        )
        
        args.conll_suffix = clean_suffixs(args.conll_suffix)
        args.dpd_suffix = clean_suffixs(args.dpd_suffix)
        args.probs_suffix = clean_suffixs(args.probs_suffix)

        syntax_label_dict = [] 
        if os.path.exists(os.path.join(paths[0], "dict.{}.txt".format("label"))):  # 单句法模式
            syntax_label_dict.append(cls.load_syntax_label_dictionary(
                os.path.join(paths[0], "dict.{}.txt".format("label"))
            )) 
        else:
            i = 0
            while os.path.exists(os.path.join(paths[0], "dict.{}.txt".format(f"label{i}"))):  # 异构句法模式
                syntax_label_dict.append(cls.load_syntax_label_dictionary(
                    os.path.join(paths[0], "dict.{}.txt".format(f"label{i}"))
                ))  # 读取句法标签
                i += 1
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))
        for i, d in enumerate(syntax_label_dict):
            logger.info("[{}] dictionary: {} types".format(f"syntax label{i}", len(d)))
        # print(syntax_label_dict)
        return cls(args, src_dict, tgt_dict, syntax_label_dict)

    @classmethod
    def load_syntax_label_dictionary(
        cls, filename
    ):
        """读取句法标签的词表
        """
        return LabelDictionary.load(filename)


    """修改记录
    Yue Zhang
    2021.12.26
    读取数据集时，额外读取：1）subword到word对齐的标签数据集；2）句法树信息数据集
    """
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            self.syntax_label_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            load_syntax=self.args.use_syntax,
            load_probs=True if self.args.syntax_encoder == "GCN" else False,
            load_subword_align=self.args.use_syntax,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
            args=self.args
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, src_nt=None, src_nt_sizes=None, src_conll_dataset=None, src_dpd_dataset=None, src_probs_dataset=None, constraints=None, syntax_type=None):
        return SyntaxEnhancedLanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            src_nt = src_nt,
            src_nt_sizes = src_nt_sizes,
            src_conll_dataset = src_conll_dataset,
            src_dpd_dataset = src_dpd_dataset,
            src_probs_dataset = src_probs_dataset,
            syntax_label_dict=self.syntax_label_dict,  # 句法标签词典
            tgt_dict=self.target_dictionary,
            constraints=constraints,
            syntax_type=syntax_type,
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
