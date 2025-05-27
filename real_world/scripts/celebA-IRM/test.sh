declare -a seed=(4)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../IRM.py --data_dir=../../data --dataset=CelebA_Blond --batch-size=48 \
-s tr_env1,tr_env2 -t te_env -a resnet18 \
--name=test3 \
--lr=0.001 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=20 \
--finetune_epochs=5 \
--phase=train

done
