data_dir=../../data/clang8_train
data_dir=../../data/wi_locness_train
src_file=$data_dir/src.txt
tgt_file=$data_dir/tgt.txt
m2_file=$data_dir/m2_reversed.txt
vanilla_parser_path=biaffine-dep-roberta-en
vanilla_parser_path=/mnt/nas_alinlp/zuyi.bzy/zhangyue/syntactic_GEC/tools/parser/biaffine-dep-roberta-en/biaffine-dep-roberta-en-gec
gopar_path=../../model/gopar/biaffine-dep-electra-en-gopar-test

# Step 1. Parse the target-side sentences in parallel GEC data by an off-the-shelf parser 
## If you find this step cost too much time, you can split the large file to several small files and predict them on multiple GPUs, and then merge the results.
python ../../src/src_gopar/parse.py $tgt_file $tgt_file.conll_predict $vanilla_parser_path

# Step 2. Extract edits by ERRANT from target-side to source-side
## If you meet this error: `OSError: [E050] Can't find model 'en'.`
## Please first run this command: `python -m spacy download en`
errant_parallel -orig $tgt_file -cor $src_file -out $m2_file
 
# Step 3. Project the target-side trees to source-side ones
python ../../src/src_gopar/convert_gec_data_to_parsing_data_english.py $tgt_file.conll_predict $m2_file $src_file.conll_converted_gopar

# Step 4. Train GOPar
## You should also re-run the 1-3 steps to generate dev/test data (BEA19-dev & CoNLL14-test)
mkdir -p $gopar_path
python -m torch.distributed.launch --nproc_per_node=8 --master_port=10000 \
       -m supar.cmds.biaffine_dep train -b -d 0,1,2,3,4,5,6,7 -c ../../src/src_gopar/configs/ptb.biaffine.dep.electra.ini -p $gopar_path/model -f char --encoder bert --bert google/electra-large-discriminator \
       --train $src_file.conll_converted_gopar \
       --dev ../../data/bea19_dev/src.txt.conll_converted_gopar \
       --test ../../data/conll14_test/src.txt.conll_converted_gopar \
       --seed 1 \
       --punct 

# Step 5. Predict source-side trees for GEC training
CoNLL_SUFFIX=conll_predict_gopar
# clang8 dataset (training stage 1)
IN_FILE=../../data/clang8_train/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=0 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

# error-coded dataset (training stage 2)
IN_FILE=../../data/error_coded_train/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=1 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

# wi-locness dataset (training stage 3)
IN_FILE=../../data/wi_locness_train/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=2 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

# bea-19-dev dataset (validation)
IN_FILE=../../data/bea19_dev/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=3 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

# bea-19-test dataset (testing)
IN_FILE=../../data/bea19_test/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=4 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

# conll-14-test dataset (testing)
IN_FILE=../../data/conll14_test/src.txt
OUT_FILE=$IN_FILE.${CoNLL_SUFFIX}
CUDA_VISIBLE_DEVICES=5 nohup python ../../src/src_gopar/parse.py $IN_FILE $OUT_FILE $gopar_path/model &

wait