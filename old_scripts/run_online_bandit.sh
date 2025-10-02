# Train online bandit
CUDA_VISIBLE_DEVICES=0 python3 train_online.py --env bandit --envs 200 --H 500 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --seed 1 --samples_per_iter 64 --num_epochs 10 --debug

# No use samples, 
for epoch in {1..10}; do
    CUDA_VISIBLE_DEVICES=0 python3 eval_online.py --env bandit --envs 200 --H 500 --dim 5 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch $epoch --n_eval 200 --seed 1
done
