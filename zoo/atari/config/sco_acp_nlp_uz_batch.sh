# envs=(
#     'AlienNoFrameskip-v4'
#     'AmidarNoFrameskip-v4'
#     'AssaultNoFrameskip-v4'
#     'AsterixNoFrameskip-v4'
#     'BankHeistNoFrameskip-v4'
#     'BattleZoneNoFrameskip-v4'
#     'ChopperCommandNoFrameskip-v4'
#     'CrazyClimberNoFrameskip-v4'
#     'DemonAttackNoFrameskip-v4'
#     'FreewayNoFrameskip-v4'
#     'FrostbiteNoFrameskip-v4'
#     'GopherNoFrameskip-v4'
#     'HeroNoFrameskip-v4'
#     'JamesbondNoFrameskip-v4'
#     'KangarooNoFrameskip-v4'
#     'KrullNoFrameskip-v4'
#     'KungFuMasterNoFrameskip-v4'
#     'PrivateEyeNoFrameskip-v4'
#     'RoadRunnerNoFrameskip-v4'
#     'UpNDownNoFrameskip-v4'
#     'PongNoFrameskip-v4'
#     'MsPacmanNoFrameskip-v4'
#     'QbertNoFrameskip-v4'
#     'SeaquestNoFrameskip-v4'
#     'BoxingNoFrameskip-v4'
#     'BreakoutNoFrameskip-v4'
# )

# one env
# env='AsterixNoFrameskip-v4'
# seed=0
# script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ && cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install ale-py autorom && AutoROM --accept-license && python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_unizero_sgement_config_batch.py --env %q --seed %d'
# script=${script/\%q/$env}
# script=${script/\%d/$seed}
# echo "The final script is: " $script

# batch env: uz表现和ez相当的15env
# envs=(
#     'AlienNoFrameskip-v4'
#     'AmidarNoFrameskip-v4'
#     'AssaultNoFrameskip-v4'
#     'BankHeistNoFrameskip-v4'
#     'BattleZoneNoFrameskip-v4'
#     'ChopperCommandNoFrameskip-v4'
#     'FreewayNoFrameskip-v4'
#     'FrostbiteNoFrameskip-v4'
#     'JamesbondNoFrameskip-v4'
#     'KangarooNoFrameskip-v4'
#     'KrullNoFrameskip-v4'
#     'PrivateEyeNoFrameskip-v4'
#     'MsPacmanNoFrameskip-v4'
#     'SeaquestNoFrameskip-v4'
#     'BoxingNoFrameskip-v4'
# )

batch env: uz表现不如ez的10env+pong
envs=(
    # 'PongNoFrameskip-v4'
    'QbertNoFrameskip-v4'
    'AsterixNoFrameskip-v4'
    'CrazyClimberNoFrameskip-v4'
    'DemonAttackNoFrameskip-v4'
    'UpNDownNoFrameskip-v4'
    'BreakoutNoFrameskip-v4'
    'GopherNoFrameskip-v4'
    'HeroNoFrameskip-v4'
    'KungFuMasterNoFrameskip-v4'
    'RoadRunnerNoFrameskip-v4' 
)
seed=0
for env in "${envs[@]}"; do
    script='source activate base &&  export HTTPS_PROXY=http://172.16.1.135:3128/ &&  cd /mnt/afs/niuyazhe/code/LightZero && pip install -e . -i  https://pkg.sensetime.com/repository/pypi-proxy/simple/ && pip3 install gym[atari]==0.25.1 ale-py==0.7.5 autorom==0.4.2 && AutoROM --accept-license && pip install pyecharts && pip show gym && pip show ale-py && pip show autorom &&  python3 -u /mnt/afs/niuyazhe/code/LightZero/zoo/atari/config/atari_unizero_reanalyze_config_batch.py --env %q --seed %d'
	script=${script/\%q/$env}
    script=${script/\%d/$seed}
	echo "The final script is: " $script

sco acp jobs create --workspace-name=df42ac16-77cf-4cfe-a3ce-e89e317bdf20 \
    --aec2-name=ea2d41fe-274a-43b2-b562-70c0b7d396a2\
    --job-name="uz-nlayer2-brf1-10000-rbs160-rr025-bs256-pew1e-4-decay10k-fixvalueV10-fixtargetaction-500k-$env-s$seed" \
    --container-image-url='registry.cn-sh-01.sensecore.cn/basemodel-ccr/aicl-b27637a9-660e-4927:20231222-17h24m12s' \
    --training-framework=pytorch \
    --enable-mpi \
    --worker-nodes=1 \
    --worker-spec='N2lS.Ii.I60.1' \
    --storage-mount 6f8b7bf6-c313-11ed-adcf-92dd2c58bebc:/mnt/afs \
    --command="$script"
done
    # --job-name="uz-nlayer4-H10-seg8-gsl20-brf1-10000-rbs160-rr1-temp025-fixvalueV8-fixtargetaction-fixtruc1-td5-200k-$env-s$seed" \


    # --job-name="uz-nlayer4-H10-seg8-gsl20-brf1-10-rbs160-rr1-temp025-pew1e-2-$env-s$seed" \
    # --job-name="uz-nlayer4-H10-seg8-gsl20-brf1-10-rbs160-rr1-temp025-$env-s$seed" \