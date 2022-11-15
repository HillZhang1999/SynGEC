####################
# Preprocess CLang8
####################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_clang8_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_bart_np

# File path
TRAIN_SRC_FILE=../../data/clang8_train/src.txt
TRAIN_TGT_FILE=../../data/clang8_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
  echo "Apply BPE..."
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_SRC_FILE \
            --outputs $TRAIN_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_TGT_FILE \
            --outputs $TRAIN_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_SRC_FILE \
            --outputs $VALID_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_TGT_FILE \
            --outputs $VALID_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
fi

# Decode
if [ ! -f $TRAIN_SRC_FILE".bart_bpe.tok" ]; then
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $TRAIN_SRC_FILE".bart_bpe" \
          --outputs $TRAIN_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $VALID_SRC_FILE".bart_bpe" \
          --outputs $VALID_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".bart_swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bart_bpe.tok" $TRAIN_SRC_FILE".bart_swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bart_bpe.tok" $VALID_SRC_FILE".bart_swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".bart_swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".bart_swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX probs bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX probs bart


cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src

if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TRAIN_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
  python ../../utils/calculate_dependency_distance.py $VALID_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/train.dpd.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/valid.dpd.src

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/train.probs.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/valid.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --trainpref $PROCESSED_DIR/train.bpe \
       --validpref $PROCESSED_DIR/valid.bpe \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../pretrained_weights/dict.txt \
       --tgtdict ../../pretrained_weights/dict.txt

echo "Finished!"

################################
# Preprocess Error-Coded-Dataset
################################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_error_coded_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_bart_np

# File path
TRAIN_SRC_FILE=../../data/error_coded_train/src.txt
TRAIN_TGT_FILE=../../data/error_coded_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
  echo "Apply BPE..."
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_SRC_FILE \
            --outputs $TRAIN_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_TGT_FILE \
            --outputs $TRAIN_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_SRC_FILE \
            --outputs $VALID_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_TGT_FILE \
            --outputs $VALID_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
fi

# Decode
if [ ! -f $TRAIN_SRC_FILE".bart_bpe.tok" ]; then
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $TRAIN_SRC_FILE".bart_bpe" \
          --outputs $TRAIN_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $VALID_SRC_FILE".bart_bpe" \
          --outputs $VALID_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".bart_swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bart_bpe.tok" $TRAIN_SRC_FILE".bart_swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bart_bpe.tok" $VALID_SRC_FILE".bart_swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".bart_swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".bart_swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX probs bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX probs bart


cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src

if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TRAIN_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
  python ../../utils/calculate_dependency_distance.py $VALID_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/train.dpd.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/valid.dpd.src

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/train.probs.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/valid.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --trainpref $PROCESSED_DIR/train.bpe \
       --validpref $PROCESSED_DIR/valid.bpe \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../pretrained_weights/dict.txt \
       --tgtdict ../../pretrained_weights/dict.txt

echo "Finished!"

#######################
# Preprocess Wi+Locness
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_wi_locness_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_bart_np

# File path
TRAIN_SRC_FILE=../../data/wi_locness_train/src.txt
TRAIN_TGT_FILE=../../data/wi_locness_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
  echo "Apply BPE..."
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_SRC_FILE \
            --outputs $TRAIN_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TRAIN_TGT_FILE \
            --outputs $TRAIN_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_SRC_FILE \
            --outputs $VALID_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $VALID_TGT_FILE \
            --outputs $VALID_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
fi

# Decode
if [ ! -f $TRAIN_SRC_FILE".bart_bpe.tok" ]; then
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $TRAIN_SRC_FILE".bart_bpe" \
          --outputs $TRAIN_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $VALID_SRC_FILE".bart_bpe" \
          --outputs $VALID_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".bart_swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bart_bpe.tok" $TRAIN_SRC_FILE".bart_swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bart_bpe.tok" $VALID_SRC_FILE".bart_swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bart_bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bart_bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".bart_swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".bart_swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX probs bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX probs bart


cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src

if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TRAIN_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
  python ../../utils/calculate_dependency_distance.py $VALID_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/train.dpd.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/valid.dpd.src

cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/train.probs.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/valid.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --trainpref $PROCESSED_DIR/train.bpe \
       --validpref $PROCESSED_DIR/valid.bpe \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../pretrained_weights/dict.txt \
       --tgtdict ../../pretrained_weights/dict.txt

echo "Finished!"


#######################
# Preprocess CoNLL-14-Test
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_conll14_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_bart_np

# File path
TEST_SRC_FILE=../../data/conll14_test/src.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
  echo "Apply BPE..."
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TEST_SRC_FILE \
            --outputs $TEST_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TEST_TGT_FILE \
            --outputs $TEST_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
fi

# Decode
if [ ! -f $TEST_SRC_FILE".bart_bpe.tok" ]; then
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $TEST_SRC_FILE".bart_bpe" \
          --outputs $TEST_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
fi

# Subword Align
if [ ! -f $TEST_SRC_FILE".bart_swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE $TEST_SRC_FILE".bart_bpe.tok" $TEST_SRC_FILE".bart_swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".bart_bpe" $PROCESSED_DIR/test.bpe.src
cp $TEST_SRC_FILE".bart_swm" $PROCESSED_DIR/test.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX probs bart


cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src

if [ ! -f $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TEST_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --only-source \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --testpref $PROCESSED_DIR/test.bpe \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../pretrained_weights/dict.txt \
       --tgtdict ../../pretrained_weights/dict.txt

echo "Finished!"

#######################
# Preprocess BEA-19-Test
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_bea19_with_syntax_bart

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_bart_np

# File path
TEST_SRC_FILE=../../data/bea19_test/src.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bart_bpe" ]; then
  echo "Apply BPE..."
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TEST_SRC_FILE \
            --outputs $TEST_SRC_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
  python ../../src/src_syngec/fairseq-0.10.2/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../../pretrained_weights/encoder.json \
            --vocab-bpe ../../pretrained_weights/vocab.bpe \
            --inputs $TEST_TGT_FILE \
            --outputs $TEST_TGT_FILE".bart_bpe" \
            --workers 60 \
            --keep-empty;
fi

# Decode
if [ ! -f $TEST_SRC_FILE".bart_bpe.tok" ]; then
  python ../../utils/multiprocessing_bpe_decoder.py \
          --encoder-json ../../pretrained_weights/encoder.json \
          --vocab-bpe ../../pretrained_weights/vocab.bpe \
          --inputs $TEST_SRC_FILE".bart_bpe" \
          --outputs $TEST_SRC_FILE".bart_bpe.tok" \
          --workers 60 \
          --keep-empty;
fi

# Subword Align
if [ ! -f $TEST_SRC_FILE".bart_swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE $TEST_SRC_FILE".bart_bpe.tok" $TEST_SRC_FILE".bart_swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".bart_bpe" $PROCESSED_DIR/test.bpe.src
cp $TEST_SRC_FILE".bart_swm" $PROCESSED_DIR/test.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX conll bart
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX probs bart


cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src

if [ ! -f $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $TEST_SRC_FILE".${CoNLL_SUFFIX}" $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --only-source \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --testpref $PROCESSED_DIR/test.bpe \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../pretrained_weights/dict.txt \
       --tgtdict ../../pretrained_weights/dict.txt

echo "Finished!"
