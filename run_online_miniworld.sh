# Train online miniworld
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 train_online.py --env miniworld --envs 1000 --H 50 --lr 0.0001 --layer 4 --head 4 --shuffle --seed 1 --samples_per_iter 32 --samples 5000 --num_epochs 10

# Evaluate
for epoch in {1..10}; do
    xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 eval_online.py --env miniworld --envs 1000 --H 50 --lr 0.0001 --layer 4 --head 4 --shuffle --epoch $epoch --seed 1
done