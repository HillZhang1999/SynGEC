# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple
import random
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    GATSyntaxGuidedTransformerEncoderLayer,
    GCNSyntaxGuidedTransformerEncoderLayer,
    ZYGCNSyntaxGuidedTransformerEncoderLayer,
    DSATransformerEncoderLayer,
    GradMultiply,
    SynGECTransformerDecoderLayer
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("syntax_enhanced_transformer")
class SyntaxEnhancedTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--alpha', type=float, metavar='D',
                            help='for DSA')
        parser.add_argument('--dual-aggregation-beta', type=float, metavar='D',
                            help='beta value for dual aggregation')
        parser.add_argument('--dual-aggregation-freeze', action='store_true',
                            help='freeze beta value for dual aggregation')
        parser.add_argument('--cross-syntax-fuse', action='store_true',
                            help='use cross attention in decoder to fuse heterogeneous syntax')
        parser.add_argument('--source-word-dropout', type=float, metavar='D', default=0.2,
                            help='dropout probability for source word dropout')
        parser.add_argument('--source-word-dropout-probs', action='store_true',
                            help='whether dropout syntactic probabilities matrix')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--gat-encoder-layers', type=int, metavar='N',
                            help='num gat encoder layers for syntax-enhanced transformer encoder')
        parser.add_argument('--gat-attention-heads', type=int, metavar='N',
                            help='num gat encoder attention headsfor syntax-enhanced transformer encoder')
        parser.add_argument('--gated-sum', action='store_true',
                            help='use gate mechanism for fusing')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--only-dsa', action='store_true')
        parser.add_argument('--only-gnn', action='store_true')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--scale-syntax-encoder-lr', type=float, metavar='D')                         
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),  # adaptive loss 是个啥？
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')  # cross attention 是个啥？
                            # When attention is performed on queries, keys and values generated from same embedding is called self attention.
                            # When attention is performed on queries generated from one embedding and keys and values generated from another embeddings is called cross attention
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')  # 主要用于模型轻量化
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')  # 主要用于模型压缩
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        syntax_enhanced_transformer(args)

        if args.encoder_layers_to_keep:  # 可以只使用一些特定的layer
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:  # 控制输入输出序列的最大长度
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        
        syntax_label_dict = getattr(task, "syntax_label_dict", None)  # 句法标签词典

        if args.share_all_embeddings:  # Transformer的Encoder和Decoder可以共享词嵌入矩阵，减少参数量，效果也不差（前提：共享词表）
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        syntax_label_embed_tokens = None
        if args.use_syntax:
            if len(syntax_label_dict) > 1:
                syntax_label_embed_tokens = []
                for d in syntax_label_dict:
                    syntax_label_embed_tokens.append(cls.build_embedding(
                        args, d, args.encoder_embed_dim
                    ))
            else:
                syntax_label_embed_tokens = cls.build_embedding(
                        args, syntax_label_dict[0], args.encoder_embed_dim
                    )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, syntax_label_dict, syntax_label_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if getattr(args, "freeze_bart_parameters", False):
            for name, param in encoder.named_parameters():
                if "sentence_encoder" in name:
                    param.requires_grad = False
            for name, param in decoder.named_parameters():
                param.requires_grad = False
        return cls(args, encoder, decoder)

    def param_groups(self):
        if not self.args.use_syntax:
            params = list(
            filter(
                lambda p: p.requires_grad,
                self.parameters())
            )
        else:
            syntax_encoder_params = list(map(id, self.encoder.syntax_encoder.parameters()))
            other_params = filter(lambda p: id(p) not in syntax_encoder_params, self.parameters())
            params = [{'params': syntax_encoder_params}, {'params': other_params}]
        return params

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, syntax_label_dict, embed_rels):
        return SyntaxEnhancedTransformerEncoder(args, src_dict, embed_tokens, syntax_label_dict, embed_rels)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return SyntaxEnhancedTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        source_tokens_nt = None,  # 带有non-terminal节点的句子，用于成分句法树编码
        source_tokens_nt_lengths = None,  # 带有non-terminal节点的句子长度
        src_outcoming_arc_mask = None,  # 描述源端句法树，出弧（指向孩子节点）掩码矩阵
        src_incoming_arc_mask = None,  # 描述源端句法树，入弧（指向孩子节点）掩码矩阵
        src_dpd_matrix = None,  # 描述源端句法树，依存距离矩阵
        src_probs_matrix = None,  # 描述源端句法树，弧概率矩阵
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        """修改记录
        2022.1.24
        Yue Zhang
        发现generation需要单独取出encoder的结果，而我们之前设计了两个encoder，然后直接在Transformer中对其进行fusion的方式，不能满足fairseq的生成逻辑
        因此，我单独抽象了一个encoder，和原先一致
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, src_outcoming_arc_mask=src_outcoming_arc_mask, src_incoming_arc_mask=src_incoming_arc_mask, src_dpd_matrix= src_dpd_matrix, src_probs_matrix= src_probs_matrix, source_tokens_nt = source_tokens_nt, source_tokens_nt_lengths= source_tokens_nt_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )  # decoder暂时不变
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class SyntaxEnhancedTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, src_dict, embed_tokens, syntax_label_dict=None, embed_rels=None):
        self.args = args
        super().__init__(src_dict)
        self.sentence_encoder = self.build_sentence_encoder(args, src_dict, embed_tokens)
        if syntax_label_dict is not None:
            syntax_type = ["dep"] * len(syntax_label_dict) if len(self.args.syntax_type) < len(syntax_label_dict) else self.args.syntax_type
        if self.args.use_syntax:
            if len(syntax_label_dict) > 1:
                self.syntax_encoder = nn.ModuleList([])  # 加入异构句法的支持，每种句法分别用一个GCN编码，最后将所有表示拼接
                for i, (d, e) in enumerate(zip(syntax_label_dict, embed_rels)):
                    self.syntax_encoder.append(self.build_syntax_guided_encoder(args, d, e, syntax_type[i]))  # 创建syntax encoder模块
            else:
                self.syntax_encoder = self.build_syntax_guided_encoder(args, syntax_label_dict[0], embed_rels, syntax_type[0])
            if getattr(args, 'gated_sum', False):
                self.quant_noise = getattr(args, "quant_noise_pq", 0)
                self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
                self.fc = self.build_fc(2 * args.encoder_embed_dim, 1, self.quant_noise, self.quant_noise_block_size)  # 对应论文公式(7)的FFN层

    def build_sentence_encoder(self, args, dictionary, embed):
        return SyntaxEnhancedSentenceTransformerEncoder(args, dictionary, embed)

    def build_syntax_guided_encoder(self, args, dictionary, embed, syntax_type="dep"):
        return SyntaxEnhancedSyntaxGuidedTransformerEncoder(args, dictionary, embed, syntax_type)

    def build_fc(self, input_dim, output_dim, q_noise=None, qn_block_size=None):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    # @staticmethod
    def dual_aggregation(self, o1, o2, beta=0.5):
        if self.args.cross_syntax_fuse:
            sentence_encoder_out = o1.encoder_out
            final_encoder_out = []
            for syntax_encoder_out in o2:
                final_encoder_out.append(beta * sentence_encoder_out + (1.0 - beta) * syntax_encoder_out.encoder_out)
            return EncoderOut(
                encoder_out=final_encoder_out,  # T x B x C
                encoder_padding_mask=o1.encoder_padding_mask,  # B x T
                encoder_embedding=o1.encoder_embedding,  # B x T x C
                encoder_states=o1.encoder_states,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )
        else:
            sentence_encoder_out = o1.encoder_out
            if len(o2) == 1:
                syntax_encoder_out = o2[0].encoder_out
            elif len(o2) == 2:
                gate_value = torch.sigmoid(self.fc(torch.cat([o2[0].encoder_out, o2[1].encoder_out], -1))) if getattr(self.args, 'gated_sum', False) else 0.5
                syntax_encoder_out = gate_value * o2[0].encoder_out + (1 - gate_value) * o2[0].encoder_out
            else:
                syntax_encoder_out = o2[0].encoder_out
                for i in range(1, len(o2)):
                    syntax_encoder_out = syntax_encoder_out + o2[i].encoder_out
            if self.training and self.args.scale_syntax_encoder_lr is not None:
                syntax_encoder_out = GradMultiply.apply(syntax_encoder_out, self.args.scale_syntax_encoder_lr)
            final_encoder_out = beta * sentence_encoder_out + (1.0 - beta) * syntax_encoder_out
            return EncoderOut(
                encoder_out=final_encoder_out,  # T x B x C
                encoder_padding_mask=o1.encoder_padding_mask,  # B x T
                encoder_embedding=o1.encoder_embedding,  # B x T x C
                encoder_states=o1.encoder_states,  # List[T x B x C]
                src_tokens=None,
                src_lengths=None,
            )


    def forward(
        self,
        src_tokens,
        src_lengths,
        source_tokens_nt = None, 
        source_tokens_nt_lengths = None,
        src_outcoming_arc_mask = None,  # 描述源端句法树，出弧（指向孩子节点）掩码矩阵
        src_incoming_arc_mask = None,  # 描述源端句法树，入弧（指向孩子节点）掩码矩阵
        src_dpd_matrix = None,  # 描述源端句法树，依存距离矩阵
        src_probs_matrix = None,  # 描述源端句法树，依存弧概率矩阵
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        if (not self.args.use_syntax) or self.args.only_gnn:
            sentence_encoder_out = self.sentence_encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, src_dpd_matrix=None
            )  # B * L * D
        else:
            sentence_encoder_out = self.sentence_encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, src_dpd_matrix=src_dpd_matrix[0]
            )  # B * L * D

        if self.args.only_dsa:
            return sentence_encoder_out

        if hasattr(self, "syntax_encoder") and isinstance(self.syntax_encoder, torch.nn.modules.container.ModuleList):
            assert len(self.syntax_encoder) == len(src_outcoming_arc_mask)
        # 这里需要再设计一个syntax-encoder，将encoder_out传入该encoder，得到syntax-encoder结果，然后进行一个dual aggregation，得到encoder的最终结果，传入decoder
        if self.args.use_syntax:
            syntax_encoder_out_all = []
            if self.training and self.args.scale_syntax_encoder_lr is not None:
                h = GradMultiply.apply(sentence_encoder_out.encoder_out, 1 / self.args.scale_syntax_encoder_lr)
            else:
                h = sentence_encoder_out.encoder_out
            h_t = h.transpose(0, 1)
            if isinstance(self.syntax_encoder, torch.nn.modules.container.ModuleList):
                for i in range(len(self.syntax_encoder)):
                    if self.args.syntax_type[i] == "dep":
                        assert src_incoming_arc_mask[i].shape[1] == src_tokens.shape[1]  # 确保tokens和掩码矩阵是同一个batch
                        syntax_encoder_out_all.append(self.syntax_encoder[i](
                            src_tokens, h=h, src_outcoming_arc_mask=src_outcoming_arc_mask[i], src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths, src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens
                        ))
                    else:  # TODO:暂时不支持成分句法的异构编码，即成分句法树最多只能引入一个，但是可以和依存句法树同时使用
                        if source_tokens_nt is not None:  # 把non-terminal节点加到句尾的scheme
                            h_new, _ = self.sentence_encoder.forward_embedding(source_tokens_nt)  # 原始token编码从sentence encoder来，non-terminal节点重新过一遍embedding层；
                            B, T, D = h_new.shape
                            # 加入non-terminal节点的编码
                            for j in range(B):  # 遍历每个sentence，除了Padding（不重要）和
                                L1 = src_lengths[j].item()
                                L2 = source_tokens_nt_lengths[j].item()
                                h_new[j, -L2:-(L2-L1+1),:] = h_t[j, -L1:-1, :]  # 正常token的表示需要用sentence encoder的编码结果
                            h_new[:, -1, :] = h_t[:, -1, :]  # EOS表示也用sentence encoder的编码结果
                            h_new = h_new.transpose(0, 1)
                            # 丢弃non-terminal节点
                            res = self.syntax_encoder[i](
                                src_tokens, h=h_new, src_outcoming_arc_mask=src_outcoming_arc_mask[i], src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths, src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens,
                            )
                            h_syn = res.encoder_out  # T_NT x B x D
                            h_refine = h.clone().detach()  # T x B x D，记住要detach，否则会连带计算图一起复制
                            for j in range(h_refine.shape[1]):
                                L1 = src_lengths[j].item()
                                L2 = source_tokens_nt_lengths[j].item()
                                h_refine[-L1:-1, j, :] = h_syn[-L2:-(L2-L1+1), j,:]  # 首先还原terminal节点
                            h_refine[-1, :, :] = h_syn[-1, :, :]  # 再还原EOS表示
                            res = EncoderOut(
                                encoder_out=h_refine,  # T x B x C
                                encoder_padding_mask=res.encoder_padding_mask,  # B x T
                                encoder_embedding=None,  # B x T x C
                                encoder_states=res.encoder_states,  # List[T x B x C]
                                src_tokens=None,
                                src_lengths=None,
                            )
                            syntax_encoder_out_all.append(res) 
            else:
                if self.args.syntax_type[0] == "dep":
                    assert src_incoming_arc_mask[0].shape[1] == src_tokens.shape[1]  # 确保tokens和掩码矩阵是同一个batch
                    syntax_encoder_out_all.append(self.syntax_encoder(
                            src_tokens, h=h, src_outcoming_arc_mask=src_outcoming_arc_mask[0], src_incoming_arc_mask=src_incoming_arc_mask[0], src_lengths=src_lengths, src_probs_matrix=src_probs_matrix[0], return_all_hiddens=return_all_hiddens
                        ))
                else:
                    if source_tokens_nt is not None:  # 把non-terminal节点加到句尾的scheme
                        i = 0
                        h_new, _ = self.sentence_encoder.forward_embedding(source_tokens_nt)  # 原始token编码从sentence encoder来，non-terminal节点重新过一遍embedding层；
                        B, T, D = h_new.shape
                        # 加入non-terminal节点的编码
                        for j in range(B):  # 遍历每个sentence，除了Padding（不重要）和
                            L1 = src_lengths[j].item()
                            L2 = source_tokens_nt_lengths[j].item()
                            h_new[j, -L2:-(L2-L1+1),:] = h_t[j, -L1:-1, :]  # 正常token的表示需要用sentence encoder的编码结果
                        h_new[:, -1, :] = h_t[:, -1, :]  # EOS表示也用sentence encoder的编码结果
                        h_new = h_new.transpose(0, 1)
                        # 丢弃non-terminal节点
                        res = self.syntax_encoder(
                            src_tokens, h=h_new, src_outcoming_arc_mask=src_outcoming_arc_mask[i], src_incoming_arc_mask=src_incoming_arc_mask[i], src_lengths=src_lengths, src_probs_matrix=src_probs_matrix[i], return_all_hiddens=return_all_hiddens,
                        )
                        h_syn = res.encoder_out  # T_NT x B x D
                        h_refine = h.clone().detach()  # T x B x D，记住要detach，否则会连带计算图一起复制
                        for j in range(h_refine.shape[1]):
                            L1 = src_lengths[j].item()
                            L2 = source_tokens_nt_lengths[j].item()
                            h_refine[-L1:-1, j, :] = h_syn[-L2:-(L2-L1+1), j,:]  # 首先还原terminal节点
                        h_refine[-1, :, :] = h_syn[-1, :, :]  # 再还原EOS表示
                        res = EncoderOut(
                            encoder_out=h_refine,  # T x B x C
                            encoder_padding_mask=res.encoder_padding_mask,  # B x T
                            encoder_embedding=None,  # B x T x C
                            encoder_states=res.encoder_states,  # List[T x B x C]
                            src_tokens=None,
                            src_lengths=None,
                        )
                        syntax_encoder_out_all.append(res)  
            encoder_out = self.dual_aggregation(sentence_encoder_out, syntax_encoder_out_all, self.args.dual_aggregation_beta)  # 聚集操作，保留原始sentence encoder的信息
            return encoder_out
        else:
            return sentence_encoder_out
 
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        if encoder_out.encoder_out is None:
            new_encoder_out = (encoder_out.encoder_out)
        elif isinstance(encoder_out.encoder_out, list):
            new_encoder_out = ([eo.index_select(1, new_order) for eo in encoder_out.encoder_out])
        else:
            new_encoder_out = (encoder_out.encoder_out.index_select(1, new_order))

        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.sentence_encoder.embed_positions is None:
            return self.sentence_encoder.max_source_positions
        return min(self.sentence_encoder.max_source_positions, self.sentence_encoder.embed_positions.max_positions)

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     return self.sentence_encoder.upgrade_state_dict_named(state_dict, name)


class SyntaxEnhancedSentenceTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding

    编码句子token信息，和传统的Transformer Encoder一致
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.args = args
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        self.src_dropout = args.source_word_dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)  # if full mask then embed_scale must be 1

        # print(self.padding_idx)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return DSATransformerEncoderLayer(args)

    
    """修改记录
    Yue Zhang
    2021.12.27
    源词丢弃策略，提高模型泛化性
    """
    def SRC_dropout(self, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens * (1 / keep_prob)


    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        if self.training:
            token_embedding = self.SRC_dropout(token_embedding, self.src_dropout)
        x = embed = self.embed_scale * token_embedding 
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_dpd_matrix = None,  # 描述源端句法树，依存距离矩阵
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        """修改记录
        Yue Zhang
        2022.1.29
        
        为Sentence Encoder添加DSA机制（CIKM21）
        """

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # print(x.shape)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask, layer_num=idx, src_dpd_matrix=src_dpd_matrix)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

"""修改记录
Yue Zhang
2021.12.29
句法诱导的Encoder，复现自论文《A Syntax-Guided Grammatical Error Correction Model with Dependency Tree Correction》
"""
class SyntaxEnhancedSyntaxGuidedTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding

    额外编码句法信息
    """

    def __init__(self, args, dictionary, embed_tokens, syntax_type="dep"):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.syntax_type = syntax_type
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop
        
        self.encoder_type = args.syntax_encoder
        self.src_dropout = args.source_word_dropout if getattr(args, "source_word_dropout_probs", False) else 0

        embed_dim = embed_tokens.embedding_dim  # 这里主要是对Relation做Embedding
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)  # if full mask then embed_scale must be 1

        self.embed_positions = None  # 关系嵌入不考虑Position

        self.layernorm_embedding = None  # 关系嵌入暂时不考虑对嵌入结果做LayerNorm

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.gat_encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        if args.syntax_encoder == "GAT":
            return GATSyntaxGuidedTransformerEncoderLayer(args)
        elif args.syntax_encoder == "GCN":
            return GCNSyntaxGuidedTransformerEncoderLayer(args, self.syntax_type)
        else:
            raise NotImplementedError

    
    """修改记录
    Yue Zhang
    2021.12.27
    源词丢弃策略，提高模型泛化性
    """
    def SRC_dropout(self, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens


    def forward_label_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        """
        Yue Zhang
        2021.12.29
        获取句法label的嵌入表示
        """
        # arc_mask: bsz * seq_len * seq_len
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding 
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        # x: bsz * seq_len * seq_len * embed_dim
        return x, embed


    def forward(
        self,
        src_tokens,
        src_lengths,
        h,  # Sentence Encoder 的输出, T x B x C
        src_outcoming_arc_mask = None,  # 描述源端句法树，出弧（指向孩子节点）掩码矩阵
        src_incoming_arc_mask = None,  # 描述源端句法树，入弧（指向孩子节点）掩码矩阵
        src_probs_matrix = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # B X T X T X C
        src_outcoming_arc_x, src_outcoming_arc_embed = self.forward_label_embedding(src_outcoming_arc_mask)  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        src_incoming_arc_x, src_incoming_arc_embed = self.forward_label_embedding(src_incoming_arc_mask)  # x包含了位置嵌入信息，encoder_embedding只包含了词嵌入信息
        del src_outcoming_arc_embed
        del src_incoming_arc_embed

        # # T x B x C -> B x T x C (Todo: 继续将T放在第一维度)
        # compute padding mask
        src_arc_padding_mask = (src_outcoming_arc_mask + src_incoming_arc_mask).eq(self.padding_idx)  # B x T x T，由于一个节点不可能同时是另一个节点的父亲/孩子，因此可以合并两个mask矩阵，pad的含义是：既非父亲也非孩子的那些节点
        encoder_states = [] if return_all_hiddens else None
        if self.training:
            src_probs_matrix = self.SRC_dropout(src_probs_matrix, self.src_dropout)
        # encoder layers
        for layer in self.layers:
            if self.encoder_type == "GAT":
                if self.syntax_type == "con":  # GAT暂时不支持编码成分句法
                    raise NotImplementedError
                encoder_padding_mask = src_tokens.eq(1)  # B x T
                h = layer(h, src_outcoming_arc_x, src_incoming_arc_x, encoder_padding_mask, attn_mask=src_arc_padding_mask)
            else:
                encoder_padding_mask = src_tokens.ne(1)  # B x T
                h = layer(h, src_outcoming_arc_x, src_incoming_arc_x, encoder_padding_mask, src_probs_matrix=src_probs_matrix)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(h)

        if self.layer_norm is not None:
            h = self.layer_norm(h)  # Layer norm是否对Batch First生效？
        return EncoderOut(
            encoder_out=h,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class SyntaxEnhancedTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    # def build_decoder_layer(self, args, no_encoder_attn=False):
    #     return TransformerDecoderLayer(args, no_encoder_attn)
    
    def build_decoder_layer(self, args, no_encoder_attn=False):
        if self.args.cross_syntax_fuse:
            return SynGECTransformerDecoderLayer(args, no_encoder_attn)
        else:
            return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("syntax_enhanced_transformer", "syntax_enhanced_transformer")
def syntax_enhanced_transformer(args):
    args.alpha = getattr(args, "alpha'", 1.0)

    args.only_dsa = getattr(args, "only_dsa", False)
    args.only_gnn = getattr(args, "only_gnn", False)
    args.cross_syntax_fuse = getattr(args, "cross_syntax_fuse", False)
    args.dual_aggregation_beta = getattr(args, "dual_aggregation_beta", 0.5)
    args.scale_syntax_encoder_lr = getattr(args, "scale_syntax_encoder_lr", None)
    args.dual_aggregation_freeze = getattr(args, "dual_aggregation_freeze", True)
    args.source_word_dropout_probs = getattr(args, "source_word_dropout_probs", False)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.gat_encoder_layers = getattr(args, "gat_encoder_layers", 3)
    args.gat_attention_heads = getattr(args, "gat_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.source_word_dropout = getattr(args, "source_word_dropout", 0.2)  # To alleviate over-fitting (2018 NAACL)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.gated_sum = getattr(args, "gated_sum", False)

# default parameters used in tensor2tensor implementation
@register_model_architecture("syntax_enhanced_transformer", "syntax_enhanced_transformer_big")
def syntax_enhanced_transformer_big(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.gat_attention_heads = getattr(args, "gat_attention_heads", 4)
    args.gat_encoder_layers = getattr(args, "gat_encoder_layers", 3)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    syntax_enhanced_transformer(args)