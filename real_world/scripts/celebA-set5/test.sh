declare -a seed=(4)
declare -a device=(1)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=CelebA_Blond --batch-size=48 \
-s tr_env1,tr_env2 -t te_env -a resnet50 \
--name=test2 \
--z_dim=64 --hidden_dim=256 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=5 \
--finetune_epochs=1 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
