
ALGO=crossq
EXP_ID="37"

if [ "$#" -ge 1 ]; then
    ALGO=$1
fi

if [ "$#" -ge 2 ]; then
    EXP_ID="$2"
fi

if [ "$EXP_ID" != "" ]; then
    EXP_ID="--exp-id $EXP_ID"
fi

python enjoy_sbx.py --algo=$ALGO --env=sigmaban-standup-v0 -f logs/ --load-best $EXP_ID --n-timesteps=1000000

