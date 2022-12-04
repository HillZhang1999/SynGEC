####################
# Preprocess HSK+Lang8
####################

FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
PROCESSED_DIR=../../preprocess/chinese_hsk+lang8_baseline

WORKER_NUM=32
DICT_SIZE=32000

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

echo "Preprocess..."
mkdir -p $PROCESSED_DIR/bin

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang tgt \
       --user-dir ../../src/src_syngec/syngec_model \
       --task syntax-enhanced-translation \
       --trainpref $PROCESSED_DIR/train.char \
       --validpref $PROCESSED_DIR/valid.char \
       --destdir $PROCESSED_DIR/bin \
       --workers $WORKER_NUM \
       --labeldict ../../data/dicts/syntax_label_gec.dict \
       --srcdict ../../data/dicts/chinese_vocab.count.txt \
       --tgtdict ../../data/dicts/chinese_vocab.count.txt

echo "Finished!"
