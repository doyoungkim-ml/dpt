# Train online darkroom
python3 train_online.py --env darkroom_heldout --envs 10000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --seed 1 --samples_per_iter 64 --num_epochs 10

# Evaluate
for epoch in {1..10}; do
    python3 eval_online.py --env darkroom_heldout --envs 10000 --H 100 --dim 10 --lr 0.001 --layer 4 --head 4 --shuffle --epoch $epoch --seed 1
done