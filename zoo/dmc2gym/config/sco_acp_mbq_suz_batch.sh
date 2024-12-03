# sco_acp_nlp_suz_batch.sh

# test 5 envs
# envs=(
#     'cartpole-swingup'
#     'cheetah-run'
#     'hopper-hop'
#     'walker-walk'
#     'humanoid-run'
# )

# 20env + humanoid
# envs=(
#     # 'acrobot-swingup'
#     # 'cartpole-balance'
#     # 'cartpole-balance_sparse'
#     # 'cartpole-swingup'
#     # 'cartpole-swingup_sparse'
#     # 'cheetah-run'
#     # "ball_in_cup-catch"
#     "finger-spin"
#     "finger-turn_easy"
#     "finger-turn_hard"
#     # 'hopper-hop'
#     # 'hopper-stand'
#     # 'pendulum-swingup'
#     # 'quadruped-run'
#     # 'quadruped-walk'
#     'reacher-easy'
#     'reacher-hard'
#     # 'walker-run'
#     # 'walker-stand'
#     # 'walker-walk'
#     # 'humanoid-run'
# )

# # 18env
# envs=(
#     'acrobot-swingup'
#     # 'cartpole-balance'
#     # 'cartpole-balance_sparse'
#     'cartpole-swingup'
#     'cartpole-swingup_sparse'
#     'cheetah-run'
#     "ball_in_cup-catch"
#     "finger-spin"
#     "finger-turn_easy"
#     "finger-turn_hard"
#     'hopper-hop'
#     'hopper-stand'
#     'pendulum-swingup'
#     'reacher-easy'
#     'reacher-hard'
#     'walker-run'
#     'walker-stand'
#     'walker-walk'
# )

# suz表现 不好的10env
envs=(
    'acrobot-swingup'
    'cartpole-swingup'
    'cheetah-run'
    "finger-spin"
    'hopper-hop'
    'hopper-stand'
    'pendulum-swingup'
    'walker-run'
    'walker-stand'
    'walker-walk'
)
seed=0
for env in "${envs[@]}"; do
    script='source activate base && export HTTPS_PROXY=http://172.16.1.135:3128/ && pip cache purge pip install pyecharts && && export_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/afs/niuyazhe/code/.mujoco/mujoco210/bin && cd /mnt/afs/niuyazhe/code/dmc2gym && pip install -e .  && pip uninstall mujoco_py -y && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && pip install pyecharts && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/dmc2gym/config/dmc2gym_state_suz_segment_config_batch.py --env %q --seed %d'
    script=${script/\%q/$env}
    script=${script/\%d/$seed}
    echo "The final script is: " $script

sco acp jobs create --workspace-name=fb1861da-1c6c-42c7-87ed-e08d8b314a99 \
    --aec2-name=eb37789e-90bb-418d-ad4a-19ce4b81ab0c\
    --job-name="suz-nlayer2-rr025-uniform-H5-2-leansigma-gsl20-fixvalueV9-fixtargetaction-$env-s$seed" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.1' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script" \
    --priority='highest'
done
    # --job-name="suz-nlayer2-gsl20-rr025-rbf1-10-rerbs160-$env-s$seed" \
    # script='source activate base && export HTTPS_PROXY=http://172.16.1.135:3128/ && export_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/afs/niuyazhe/code/.mujoco/mujoco210/bin && cd /mnt/afs/niuyazhe/code/dmc2gym && pip install -e .  && export MUJOCO_GL="osmesa" && pip install gym==0.22.0 && pip uninstall mujoco_py -y && pip install mujoco==3.2.1 && pip install dm-control==1.0.22  && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/dmc2gym/config/dmc2gym_state_sampled_unizero_segment_config_batch.py --env %q --seed %d'
    # --job-name="suz-nlayer2-rr01-origcollect-origctree-uniform-H5-2-$env-s$seed" \