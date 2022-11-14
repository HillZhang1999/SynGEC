####################
# Preprocess HSK+Lang8
####################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/chinese_hsk+lang8_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TRAIN_SRC_FILE=../../data/hsk+lang8_train/src.txt
TRAIN_TGT_FILE=../../data/hsk+lang8_train/tgt.txt
VALID_SRC_FILE=../../data/mucgec_dev/src.txt
VALID_TGT_FILE=../../data/mucgec_dev/tgt.txt

# apply char
if [ ! -f $TRAIN_SRC_FILE".char" ]; then
  python ../../utils/segment_bert.py <$TRAIN_SRC_FILE >$TRAIN_SRC_FILE".char"
  python ../../utils/segment_bert.py <$TRAIN_TGT_FILE >$TRAIN_TGT_FILE".char"
  python ../../utils/segment_bert.py <$VALID_SRC_FILE >$VALID_SRC_FILE".char"
  python ../../utils/segment_bert.py <$VALID_TGT_FILE >$VALID_TGT_FILE".char"
fi

# Subword Align
if [ ! -f $TRAIN_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TRAIN_SRC_FILE".char" $TRAIN_SRC_FILE".char" $TRAIN_SRC_FILE".swm"
  python ../../utils/subword_align.py $VALID_SRC_FILE".char" $VALID_SRC_FILE".char" $VALID_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TRAIN_SRC_FILE $PROCESSED_DIR/train.src
cp $TRAIN_SRC_FILE".char" $PROCESSED_DIR/train.char.src
cp $TRAIN_TGT_FILE $PROCESSED_DIR/train.tgt
cp $TRAIN_TGT_FILE".char" $PROCESSED_DIR/train.char.tgt
cp $VALID_SRC_FILE $PROCESSED_DIR/valid.src
cp $VALID_SRC_FILE".char" $PROCESSED_DIR/valid.char.src
cp $VALID_TGT_FILE $PROCESSED_DIR/valid.tgt
cp $VALID_TGT_FILE".char" $PROCESSED_DIR/valid.char.tgt

cp $TRAIN_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src
cp $VALID_SRC_FILE".swm" $PROCESSED_DIR/valid.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TRAIN_SRC_FILE $CONLL_SUFFIX probs transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $VALID_SRC_FILE $CONLL_SUFFIX probs transformer


cp $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/train.conll.src
cp $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/valid.conll.src

if [ ! -f $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $PROCESSED_DIR/train.conll.src $PROCESSED_DIR/train.swm.src $TRAIN_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
  python ../../utils/calculate_dependency_distance.py $PROCESSED_DIR/valid.conll.src $PROCESSED_DIR/valid.swm.src $VALID_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
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
       --trainpref $PROCESSED_DIR/train.char \
       --validpref $PROCESSED_DIR/valid.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../data/dicts/chinese_vocab.count.txt \
       --tgtdict ../../data/dicts/chinese_vocab.count.txt

echo "Finished!"


####################
# Preprocess MuCGEC
####################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/chinese_mucgec_with_syntax_transformer

WORKER_NUM=32
DICT_SIZE=32000
CoNLL_SUFFIX=conll_predict_gopar
CoNLL_SUFFIX_PROCESSED=conll_predict_gopar_np

# File path
TEST_SRC_FILE=../../data/mucgec_test/src.txt

# apply char
if [ ! -f $TEST_SRC_FILE".char" ]; then
  python ../../utils/segment_bert.py <$TEST_SRC_FILE >$TEST_SRC_FILE".char"
fi

# Subword Align
if [ ! -f $TEST_SRC_FILE".swm" ]; then
  echo "Align subwords and words..."
  python ../../utils/subword_align.py $TEST_SRC_FILE".char" $TEST_SRC_FILE".char" $TEST_SRC_FILE".swm"
fi

# fairseq preprocess
mkdir -p $PROCESSED_DIR
cp $TEST_SRC_FILE $PROCESSED_DIR/train.src
cp $TEST_SRC_FILE".char" $PROCESSED_DIR/train.char.src
cp $TEST_SRC_FILE".swm" $PROCESSED_DIR/train.swm.src

# syntax specific
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX conll transformer
python ../../utils/syntax_information_reprocess.py $TEST_SRC_FILE $CONLL_SUFFIX probs transformer

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}" $PROCESSED_DIR/test.conll.src

if [ ! -f $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" ]; then
  echo "Calculate dependency distance..."
  python ../../utils/calculate_dependency_distance.py $PROCESSED_DIR/test.conll.src $PROCESSED_DIR/test.swm.src $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd"
fi

cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.dpd" $PROCESSED_DIR/test.dpd.src
cp $TEST_SRC_FILE".${CoNLL_SUFFIX_PROCESSED}.probs" $PROCESSED_DIR/test.probs.src

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --only-source \
       --testpref $PROCESSED_DIR/test.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --conll-suffix conll \
       --swm-suffix swm \
       --dpd-suffix dpd \
       --probs-suffix probs \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../data/dicts/chinese_vocab.count.txt \
       --tgtdict ../../data/dicts/chinese_vocab.count.txt

echo "Finished!"
