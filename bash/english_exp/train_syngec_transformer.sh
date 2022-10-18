####################
# Train Baseline
####################

SEED=2022
FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR_STAGE1=../../model/english_transformer_baseline/$SEED/stage1
MODEL_DIR_STAGE2=../../model/english_transformer_baseline/$SEED/stage2
MODEL_DIR_STAGE3=../../model/english_transformer_baseline/$SEED/stage3
PROCESSED_DIR_STAGE1=../../preprocess/english_clang8_with_syntax_transformer
PROCESSED_DIR_STAGE2=../../preprocess/english_error_coded_with_syntax_transformer
PROCESSED_DIR_STAGE3=../../preprocess/english_wi_locness_with_syntax_transformer
FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq

mkdir -p $MODEL_DIR_STAGE1

mkdir -p $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE1/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-04 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE1}/nohup.log 2>&1 &

wait

mkdir -p $MODEL_DIR_STAGE2

mkdir -p $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE2/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE2/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE2

# Transformer-base-setting stage 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE2/bin \
    --save-dir $MODEL_DIR_STAGE2 \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --finetune-from-model $MODEL_DIR_STAGE1/checkpoint_best.pt \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-05 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE2}/nohup.log 2>&1 &

wait

mkdir -p $MODEL_DIR_STAGE3

mkdir -p $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE3/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE3/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE3

# Transformer-base-setting stage 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE3/bin \
    --save-dir $MODEL_DIR_STAGE3 \
    --user-dir ../../src/src_syngec/syngec_model \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --finetune-from-model $MODEL_DIR_STAGE2/checkpoint_best.pt \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-05 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE3}/nohup.log 2>&1 &

wait

####################
# Train SynGEC
####################

SEED=2022
FAIRSEQ_CLI_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq_cli
MODEL_DIR_STAGE1=../../model/english_transformer_syngec/$SEED/stage1
MODEL_DIR_STAGE2=../../model/english_transformer_syngec/$SEED/stage2
MODEL_DIR_STAGE3=../../model/english_transformer_syngec/$SEED/stage3
PROCESSED_DIR_STAGE1=../../preprocess/english_clang8_with_syntax_transformer
PROCESSED_DIR_STAGE2=../../preprocess/english_error_coded_with_syntax_transformer
PROCESSED_DIR_STAGE3=../../preprocess/english_wi_locness_with_syntax_transformer
FAIRSEQ_PATH=../../src/src_syngec/fairseq-0.10.2/fairseq

mkdir -p $MODEL_DIR_STAGE1

mkdir -p $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE1/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE1/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE1/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE1

# Transformer-base-setting stage 1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE1/bin \
    --save-dir $MODEL_DIR_STAGE1 \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-04 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE1}/nohup.log 2>&1 &

wait

mkdir -p $MODEL_DIR_STAGE2

mkdir -p $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE2/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE2/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE2/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE2

# Transformer-base-setting stage 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE2/bin \
    --save-dir $MODEL_DIR_STAGE2 \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --finetune-from-model $MODEL_DIR_STAGE1/checkpoint_best.pt \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-05 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE2}/nohup.log 2>&1 &

wait

mkdir -p $MODEL_DIR_STAGE3

mkdir -p $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_PATH $MODEL_DIR_STAGE3/src

cp -r $FAIRSEQ_CLI_PATH $MODEL_DIR_STAGE3/src

cp -r ../../src/src_syngec/syngec_model $MODEL_DIR_STAGE3/src

cp ./train_syngec_transformer.sh $MODEL_DIR_STAGE3

# Transformer-base-setting stage 3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -u $FAIRSEQ_CLI_PATH/train.py $PROCESSED_DIR_STAGE3/bin \
    --save-dir $MODEL_DIR_STAGE3 \
    --user-dir ../../src/src_syngec/syngec_model \
    --use-syntax \
    --only-gnn \
    --syntax-encoder GCN \
    --task syntax-enhanced-translation \
    --arch syntax_enhanced_transformer \
    --skip-invalid-size-inputs-valid-test \
    --finetune-from-model $MODEL_DIR_STAGE2/checkpoint_best.pt \
    --max-source-positions 64 \
    --max-target-positions 64 \
    --max-tokens 2048 \
    --optimizer adam \
    --update-freq 4 \
    --lr 5e-05 \
    -s src \
    -t tgt \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 60 \
    --share-all-embeddings \
    --adam-betas '(0.9,0.98)' \
    --log-format tqdm \
    --find-unused-parameters \
    --fp16 \
    --keep-last-epochs 10 \
    --patience 10 \
    --seed $SEED \
    >${MODEL_DIR_STAGE3}/nohup.log 2>&1 &

wait
