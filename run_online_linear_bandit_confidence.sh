# Train online linear bandit with confidence
python3 train_online_confidence.py --env linear_bandit --envs 10000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --seed 1 --samples_per_iter 64 --samples 20000 --num_epochs 10 --confidence_start 0.3

# Evaluate
for epoch in {1..10}; do
    python3 eval_online.py --env linear_bandit --envs 10000 --H 200 --dim 10 --lin_d 2 --var 0.3 --cov 0.0 --lr 0.0001 --layer 4 --head 4 --epoch $epoch --n_eval 200 --seed 1 --confidence
done
