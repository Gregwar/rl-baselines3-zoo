#!/bin/bash

PROJECT=sigmaban-standup
ENV=sigmaban-standup-v0
ALGO=td3
TAGS=standup

killall -9 python

nohup python train.py --env=$ENV --algo=$ALGO --track --wandb-project-name $PROJECT \
    --vec-env=subproc --tags $TAGS $HOSTNAME > $HOSTNAME &

tail -f $HOSTNAME.out

