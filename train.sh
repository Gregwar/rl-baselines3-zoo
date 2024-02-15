#!/bin/bash

PROJECT=sigmaban-standup
ENV=sigmaban-standup-v0
ALGO=td3
TAGS=standup
EXP_ID=`python next_run_id.py $ALGO $ENV`

killall -9 python

nohup python train.py --env=$ENV --algo=$ALGO --track --wandb-project-name $PROJECT \
    --vec-env=subproc -tags exp-$EXP_ID $TAGS $HOSTNAME > $HOSTNAME.out &

tail -f $HOSTNAME.out

