#!/bin/bash

# kge start ./local/configs/toy-ensemble-train.yaml --job.device cpu
# kge resume ./local/experiments/20211115-081548-toy-ensemble-train/ --job.device cpu
# kge valid ./local/experiments/20211115-081548-toy-ensemble-train/
kge valid ./local/pretraining/fb15k-237/transe/ --job.device cpu