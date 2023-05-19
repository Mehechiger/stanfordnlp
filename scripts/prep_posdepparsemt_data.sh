#!/usr/bin/env bash
#
# Prepare data for training and evaluating parsers. Run as:
#   ./prep_depparse_data.sh TREEBANK ANNOTATIONS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT) and ANNOTATIONS the kept annotations.
# This script assumes UDBASE and POSDEPPARSEMT_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

treebank=$1; shift
original_short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $original_short | sed -e 's#_.*##g'`

annots=$1; shift

if [ -d "$UDBASE/${treebank}_XV" ]; then
    src_treebank="${treebank}_XV"
    src_short="${original_short}_xv"
else
    src_treebank=$treebank
    src_short=$original_short
fi

# path of input data to dependency parser training process
train_in_file=$POSDEPPARSEMT_DATA_DIR/${original_short}.train.in.combined
dev_in_file=$POSDEPPARSEMT_DATA_DIR/${original_short}.dev.in.combined
dev_gold_file=$POSDEPPARSEMT_DATA_DIR/${original_short}.dev.gold.conll

# handle languages requiring special batch size
batch_size=5000

if [ $treebank == 'UD_Galician-TreeGal' ]; then
    batch_size=3000
fi
echo "Using batch size $batch_size"

train_conllu=$UDBASE/$src_treebank/${src_short}-ud-train_trimmed6911.combined.$annots
dev_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev_trimmed1117.conll # gold dev
dev_gold_conllu=$UDBASE/$src_treebank/${src_short}-ud-dev_trimmed1117.combined
cp $train_conllu $train_in_file
cp $dev_conllu $dev_in_file
cp $dev_gold_conllu $dev_gold_file