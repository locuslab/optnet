#!/bin/bash

cd $(dirname $0)

runLearnD() {
  Dpenalty=$1
  ./main.py --nEpoch 50 optnet --learnD --Dpenalty $Dpenalty &
  sleep 4s
}

runLearnD 0.01
runLearnD 0.1
# runLearnD 0.5
# runLearnD 1.0
# runLearnD 10.0

# ./main.py --nEpoch 50 optnet --tvInit &

# runRelu() {
#   NHIDDEN=$1
#   OTHER=$2
#   ./main.py --nEpoch 50 relu --nHidden $NHIDDEN $OTHER &
#   sleep 4
# }

# runRelu 100
# runRelu 100 '--bn'
# runRelu 200
# runRelu 200 '--bn'
# runRelu 500
# runRelu 500 '--bn'
# runRelu 1000
# runRelu 1000 '--bn'
