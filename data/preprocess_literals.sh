#!/bin/sh

for D in *; do
    if [ -d "${D}" ] && ! [ "$D" = "preprocess" ]; then
      echo -n "Preprocessing literals in "
      echo $D
      python3 preprocess/preprocess_literals.py $D
      echo "Finished preprocesssing"
      echo
    fi
done
