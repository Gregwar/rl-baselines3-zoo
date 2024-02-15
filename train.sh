
PROJECT=sigmaban-standup
ENV=sigmaban-standup-v0
ALGO=td3

killall -9 python

nohup python train.py --env=$ENV --algo=$ALGO --track --wandb-project-name $PROJECT --vec-env=subproc &

tail -f nohup.out

