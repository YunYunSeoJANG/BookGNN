# ./scripts/train.sh: Permission denied --> run: chmod +x ./scripts/train.sh

python3 src/training.py --conv_layer LGC --neg_samp random &
wait

python3 src/training.py --conv_layer LGC --neg_samp hard &
wait

python3 src/training.py --conv_layer GAT --neg_samp random &
wait

python3 src/training.py --conv_layer GAT --neg_samp hard &
wait

python3 src/training.py --conv_layer SAGE --neg_samp random &
wait

python3 src/training.py --conv_layer SAGE --neg_samp hard &
wait