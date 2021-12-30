#!/bin/bash

# kge resume ./local/experiments/20211115-081548-toy-ensemble-train/ --job.device cpu
# kge valid ./local/experiments/20211115-081548-toy-ensemble-train/
# kge valid ./local/pretraining/fb15k-237/transe/ --job.device cpu
# python3 kge/cli.py start ./local/configs/toy-transe-train.yaml --job.device cpu
# python3 kge/cli.py start ./local/configs/toy-complex-train.yaml --job.device cpu
python3 kge/cli.py valid ./local/experiments/20211229-135141-toy-multilayer_perceptron-train --job.device cpu