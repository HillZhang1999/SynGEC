CUDA_DEVICE=0
BEAM=12
N_BEST=1
SEED=2022
FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR=../../model/english_bart_syngec/$SEED/stage3

CoNLL14_TEST_BIN_DIR=../../preprocess/english_conll14_with_syntax_bart/bin
BEA19_TEST_BIN_DIR=../../preprocess/english_bea19_with_syntax_bart/bin
PROCESSED_DIR=../../preprocess/english_clang8_with_syntax_bart

OUTPUT_DIR=$MODEL_DIR/results

CoNLL14_INPUT_FILE=../../data/conll14_test/src.txt
BEA19_INPUT_FILE=../../data/bea19_test/src.txt

mkdir -p $OUTPUT_DIR
cp $CoNLL14_INPUT_FILE $OUTPUT_DIR/CoNLL14.src
cp $BEA19_INPUT_FILE $OUTPUT_DIR/BEA19.src

echo "Generating CoNLL14..."
SECONDS=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --bpe gpt2 \
    --buffer-size 5000 \
    --batch-size 32 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --conll_file $CoNLL14_TEST_BIN_DIR/test.conll.src-tgt.src\
    --dpd_file $CoNLL14_TEST_BIN_DIR/test.dpd.src-tgt.src \
    --probs_file $CoNLL14_TEST_BIN_DIR/test.probs.src-tgt.src \
    --output_file $OUTPUT_DIR/CoNLL14.out.nbest \
    < $OUTPUT_DIR/CoNLL14.src

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/CoNLL14.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/CoNLL14.out
sed -i '$d' $OUTPUT_DIR/CoNLL14.out
python ../../utils/post_process_english.py $OUTPUT_DIR/CoNLL14.src $OUTPUT_DIR/CoNLL14.out $OUTPUT_DIR/CoNLL14.out.post_processed

echo "Generating BEA19..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --bpe gpt2 \
    --buffer-size 5000 \
    --batch-size 32 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --conll_file $BEA19_TEST_BIN_DIR/test.conll.src-tgt.src \
    --dpd_file $BEA19_TEST_BIN_DIR/test.dpd.src-tgt.src \
    --probs_file $BEA19_TEST_BIN_DIR/test.probs.src-tgt.src \
    --output_file $OUTPUT_DIR/BEA19.out.nbest \
    < $OUTPUT_DIR/BEA19.src

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/BEA19.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/BEA19.out
sed -i '$d' $OUTPUT_DIR/BEA19.out
python ../../utils/post_process_english.py $OUTPUT_DIR/BEA19.src $OUTPUT_DIR/BEA19.out $OUTPUT_DIR/BEA19.out.post_processed
