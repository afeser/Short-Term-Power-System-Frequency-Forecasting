#!/bin/bash


screen -S "rsyncDAI" -h 100000 -d -m bash -c "while [ 1 ]; do rsync -au /home/afeser/Documents/Yurdakul/forecasting/FrequencyForecastGit/src/* eser@10.0.2.93:/home/eser/eser/dockerWriteAccess/forecasting/eser_stuff/FrequencyForecastGit/src/; done"

