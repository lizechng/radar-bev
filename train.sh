export CUDA_VISIBLE_DEVICES=0,1,2,3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python main.py train --logdir ./exp/fine_tune_background_resume --lr 0.0001 --bsz 4
#python main.py train --logdir ./exp/mini --bsz 1 --lr 0.0001