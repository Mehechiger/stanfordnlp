#!/usr/bin/env bash
#
# Train and evaluate parser. Run as:
#   ./run_posdepparsemt.sh TREEBANK ANNOTATIONS OTHER_ARGS
# where TREEBANK is the UD treebank name (e.g., UD_English-EWT), ANNOTATIONS the kept annotations and OTHER_ARGS are additional training arguments (see parser code) or empty.
# This script assumes UDBASE and POSDEPPARSEMT_DATA_DIR are correctly set in config.sh.

source scripts/config.sh

: ${1?"Usage: $0 TREEBANK ANNOTATIONS OTHER_ARGS"}

treebank=$1; shift
annots=$1; shift
args=$@
short=`bash scripts/treebank_to_shorthand.sh ud $treebank`
lang=`echo $short | sed -e 's#_.*##g'`

train_file=${POSDEPPARSEMT_DATA_DIR}/${short}.train.in.combined
eval_file=${POSDEPPARSEMT_DATA_DIR}/${short}.dev.in.combined
output_file=${POSDEPPARSEMT_DATA_DIR}/${short}.dev.pred.conllu
gold_file=${POSDEPPARSEMT_DATA_DIR}/${short}.dev.gold.conll

if [ ! -e $train_file ]; then
    bash scripts/prep_posdepparsemt_data.sh $treebank $annots
fi

# handle languages that need reduced batch size
batch_size=5000

if [ $treebank == 'UD_Finnish-TDT' ] || [ $treebank == 'UD_Russian-Taiga' ] || [ $treebank == 'UD_Latvian-LVTB' ] || [ $treebank == 'UD_Croatian-SET' ] || [ $treebank == 'UD_Galician-TreeGal' ]; then
    batch_size=3000
fi
echo "Using batch size $batch_size"

echo "Running parser with $args..."
echo "python -m stanfordnlp.models.mt_taggerparser --wordvec_dir $WORDVEC_DIR --train_file $train_file --eval_file $eval_file --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --batch_size $batch_size --mode train $args"
exit 0
python -m stanfordnlp.models.mt_taggerparser --wordvec_dir $WORDVEC_DIR --eval_file $eval_file --output_file $output_file --gold_file $gold_file --lang $lang --shorthand $short --mode predict $args
results=`python stanfordnlp/utils/conll18_ud_eval.py -v $gold_file $output_file | head -12 | tail -n+12 | awk '{print $7}'`
echo $results $args >> ${POSDEPPARSEMT_DATA_DIR}/${short}.results
echo $short $results $args
