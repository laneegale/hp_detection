export $(grep -v '^#' .env | xargs)

python get_feats.py $MODELS $DATA_PATH $FEATURE_OUTPUT_PATH