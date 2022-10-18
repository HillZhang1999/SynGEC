CUDA_DEVICE=0
BEAM=12
N_BEST=1
SEED=2022
FAIRSEQ_DIR=../../src/src_syngec/fairseq-0.10.2/fairseq_cli

TEST_DIR=../../data/mucgec_test
MODEL_DIR=../../model/chinese_bart_syngec/$SEED/stage1
ID_FILE=$TEST_DIR/src.id
MuCGEC_TEST_BIN_DIR=../../preprocess/chinese_mucgec_with_syntax_bart/bin
PROCESSED_DIR=../../preprocess/chinese_hsk+lang8_with_syntax_bart

OUTPUT_DIR=$MODEL_DIR/results

mkdir -p $OUTPUT_DIR
cp $ID_FILE $OUTPUT_DIR/mucgec.id
cp $TEST_DIR/src.txt.char $OUTPUT_DIR/mucgec.src.char

echo "Generating MuCGEC Test..."
SECONDS=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u ${FAIRSEQ_DIR}/interactive.py $PROCESSED_DIR/bin \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${BEAM} \
    --nbest ${N_BEST} \
    -s src \
    -t tgt \
    --buffer-size 10000 \
    --batch-size 32 \
    --num-workers 12 \
    --log-format tqdm \
    --remove-bpe \
    --fp16 \
    --conll_file $MuCGEC_TEST_BIN_DIR/test.conll.src-tgt.src \
    --dpd_file $MuCGEC_TEST_BIN_DIR/test.dpd.src-tgt.src \
    --probs_file $MuCGEC_TEST_BIN_DIR/test.probs.src-tgt.src \
    --output_file $OUTPUT_DIR/mucgec.out.nbest \
    < $OUTPUT_DIR/mucgec.src.char

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

cat $OUTPUT_DIR/mucgec.out.nbest | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = ''.join([ x[i] for i in range(len(x)) if (i % ${N_BEST} == 0) ]); print(x)" | cut -f 3 > $OUTPUT_DIR/mucgec.out
sed -i '$d' $OUTPUT_DIR/mucgec.out
python ../../utils/post_process_chinese.py $OUTPUT_DIR/mucgec.src.char $OUTPUT_DIR/mucgec.out $OUTPUT_DIR/mucgec.id $OUTPUT_DIR/mucgec.out.post_processed
