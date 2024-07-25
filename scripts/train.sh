CUDA_VISIBLE_DEVICES=0 python3 src/training.py --conv_layer LGC --neg_samp random &
CUDA_VISIBLE_DEVICES=1 python3 src/training.py --conv_layer LGC --neg_samp hard &
CUDA_VISIBLE_DEVICES=2 python3 src/training.py --conv_layer GAT --neg_samp random &
CUDA_VISIBLE_DEVICES=3 python3 src/training.py --conv_layer GAT --neg_samp hard &

wait

CUDA_VISIBLE_DEVICES=0 python3 src/training.py --conv_layer SAGE --neg_samp random &
CUDA_VISIBLE_DEVICES=1 python3 src/training.py --conv_layer SAGE --neg_samp hard &