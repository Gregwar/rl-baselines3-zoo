#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

PROJECT=sigmaban-standup
ENV=sigmaban-standup-v0
ALGO=td3
TAGS=standup
EXP_ID=`python next_run_id.py $ALGO $ENV`
RUN_NAME="$HOSTNAME_${EXP_ID}_$1"

killall -9 python

nohup python train.py --env=$ENV --algo=$ALGO --track --wandb-project-name $PROJECT \
    --vec-env=subproc --run-name="$RUN_NAME" -tags exp-$EXP_ID $TAGS h-$HOSTNAME > $HOSTNAME.out &

tail -f $HOSTNAME.out

