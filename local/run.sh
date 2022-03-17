#!/bin/bash

## Train Test routine

# python3 ../kge/cli.py start ../local/configs/toy/complex_train.yaml --job.device cpu

# python3 ../kge/cli.py test ../local/pretraining/toy/transe --job.device cpu
python3 ../kge/cli.py test ../local/experiments/20220316-083348-complex_train --job.device cpu
