####################
# Preprocess CLang8
####################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_clang8_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TRAIN_SRC_FILE=../../data/clang8_train/src.txt
TRAIN_TGT_FILE=../../data/clang8_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bpe" ]; then
  echo "Apply BPE..."
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_SRC_FILE > $TRAIN_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_TGT_FILE > $TRAIN_TGT_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_SRC_FILE > $VALID_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_TGT_FILE > $VALID_TGT_FILE".bpe"
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bpe" $TRAIN_SRC_FILE".swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bpe" $VALID_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX probs transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX probs transformer


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
       --srcdict ../../data/dicts/english_vocab.count.txt \
       --tgtdict ../../data/dicts/english_vocab.count.txt

echo "Finished!"

################################
# Preprocess Error-Coded-Dataset
################################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_error_coded_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TRAIN_SRC_FILE=../../data/error_coded_train/src.txt
TRAIN_TGT_FILE=../../data/error_coded_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bpe" ]; then
  echo "Apply BPE..."
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_SRC_FILE > $TRAIN_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_TGT_FILE > $TRAIN_TGT_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_SRC_FILE > $VALID_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_TGT_FILE > $VALID_TGT_FILE".bpe"
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bpe" $TRAIN_SRC_FILE".swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bpe" $VALID_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX probs transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX probs transformer


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
       --srcdict ../../data/dicts/english_vocab.count.txt \
       --tgtdict ../../data/dicts/english_vocab.count.txt

echo "Finished!"

#######################
# Preprocess Wi+Locness
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_wi_locness_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TRAIN_SRC_FILE=../../data/wi_locness_train/src.txt
TRAIN_TGT_FILE=../../data/wi_locness_train/tgt.txt
VALID_SRC_FILE=../../data/bea19_dev/src.txt
VALID_TGT_FILE=../../data/bea19_dev/tgt.txt

# apply bpe
if [ ! -f $TRAIN_SRC_FILE".bpe" ]; then
  echo "Apply BPE..."
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_SRC_FILE > $TRAIN_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TRAIN_TGT_FILE > $TRAIN_TGT_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_SRC_FILE > $VALID_SRC_FILE".bpe"
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $VALID_TGT_FILE > $VALID_TGT_FILE".bpe"
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE $TRAIN_SRC_FILE".bpe" $TRAIN_SRC_FILE".swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE $VALID_SRC_FILE".bpe" $VALID_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".bpe" $PROCESSED_DIR/train.bpe.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".bpe" $PROCESSED_DIR/train.bpe.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".bpe" $PROCESSED_DIR/valid.bpe.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".bpe" $PROCESSED_DIR/valid.bpe.tgt

cp $TRAIN_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CoNLL_SUFFIX probs transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CoNLL_SUFFIX probs transformer


cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src

if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd1" ]; then
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
       --srcdict ../../data/dicts/english_vocab.count.txt \
       --tgtdict ../../data/dicts/english_vocab.count.txt

echo "Finished!"


#######################
# Preprocess CoNLL-14-Test
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_conll14_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TEST_SRC_FILE=../../data/conll14_test/src.txt

# apply bpe
if [ ! -f $TEST_SRC_FILE".bpe" ]; then
  echo "Apply BPE..."
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TEST_SRC_FILE > $TEST_SRC_FILE".bpe"
fi

# Subword Align
if [ ! -f $TEST_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE $TEST_SRC_FILE".bpe" $TEST_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".bpe" $PROCESSED_DIR/test.bpe.src
cp $TEST_SRC_FILE".swm" $PROCESSED_DIR/test.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX probs transformer


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
       --srcdict ../../data/dicts/english_vocab.count.txt \
       --tgtdict ../../data/dicts/english_vocab.count.txt

echo "Finished!"

#######################
# Preprocess BEA-19-Test
#######################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/english_bea19_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TEST_SRC_FILE=../../data/bea19_test/src.txt

# apply bpe
if [ ! -f $TEST_SRC_FILE".bpe" ]; then
  echo "Apply BPE..."
  subword-nmt apply-bpe -c ../../data/dicts/dict.tgt.32k_from_all_combine.txt < $TEST_SRC_FILE > $TEST_SRC_FILE".bpe"
fi

# Subword Align
if [ ! -f $TEST_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE $TEST_SRC_FILE".bpe" $TEST_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/test.src
cp $TEST_SRC_FILE".bpe" $PROCESSED_DIR/test.bpe.src
cp $TEST_SRC_FILE".swm" $PROCESSED_DIR/test.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CoNLL_SUFFIX probs transformer


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
       --srcdict ../../data/dicts/english_vocab.count.txt \
       --tgtdict ../../data/dicts/english_vocab.count.txt

echo "Finished!"
