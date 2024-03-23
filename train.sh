#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

PROJECT=sigmaban-standup
ENV=sigmaban-standup-v0
ALGO=sac
TAGS=standup
EXP_ID=`python next_run_id.py $ALGO $ENV`
RUN_NAME="${HOSTNAME}_${EXP_ID}_$1"

# killall -9 python

nohup python train.py --run-name="$RUN_NAME" \
    --env=$ENV --algo=$ALGO --track --wandb-project-name $PROJECT \
    --eval-episodes=100 --n-eval-envs=4 --vec-env=subproc \
    --eval-env-kwargs evaluation:True \
    --wandb-code-dir=../mujoco_sim/ \
    -tags exp-$EXP_ID $TAGS h-$HOSTNAME > $HOSTNAME.out &

tail -f $HOSTNAME.out

