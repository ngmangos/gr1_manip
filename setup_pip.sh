python3 -m pip install -e source/gr1_train
pip3 install matplotlib
mv -f /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py \
    /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner_cpy.py
cp -f /workspace/fourier-sim/gr1_train/algos/on_policy_runner.py \
    /workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py