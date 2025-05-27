declare -a seed=(4)
declare -a device=(7)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../ERM.py --data_dir=../../data --dataset=CelebA_Blond --batch-size=48 \
-s tr_env1,tr_env2 -t te_env -a resnet18 \
--name=test \
--lr=0.001 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=20 \
--phase=train
done
