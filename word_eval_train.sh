export CHECKPOINT_DIR=/ext3/multimodal-baby/checkpoints/lm_text_encoder_lstm_embedding_dim_32_tie_True_bias_True_dropout_i_0.0_dropout_o_0.1_batch_size_32_optimizer_AdamW_lr_0.03_weight_decay_0.03_seed_0_early_save

python3 lm_code/word_evaluation_saycam.py \
        --tokenizer=${CHECKPOINT_DIR}/word2idx.json \
        --wordbank_file="r_code/tacl_data/child_data/child_aoa.tsv" \
        --examples_file="/ext3/multimodal-baby/data/saycam_train_text.txt" \
        --min_samples=8 \
        --max_samples=512 \
        --batch_size=128 \
        --output_file=${CHECKPOINT_DIR}/surprisals_train.txt \
        --model_dir=${CHECKPOINT_DIR} \
        --model_type="lstm_saycam" \
        --save_samples="/ext3/multimodal-baby/data/saycam_train_samples_updated.pkl" \
        --min_seq_len=5
