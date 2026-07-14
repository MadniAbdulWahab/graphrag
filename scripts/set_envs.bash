# Script to set hugging face environment variables to use custom directory
## Had to be done for disk quota reasons


export HF_HOME=/home/iailab43/khanm2/ir_lab/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/iailab43/khanm2/ir_lab/.cache/huggingface
export TMPDIR=/home/iailab43/khanm2/ir_lab/.cache/tmp


echo $HF_HOME
echo $HUGGINGFACE_HUB_CACHE
echo $TMPDIR
