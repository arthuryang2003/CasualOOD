declare -a seed=(4)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main.py --data_dir=../../data --dataset=Waterbirds --batch-size=48 \
-s tr_env1,tr_env2 -t te_env -a resnet50 \
--name=test3 \
--z_dim=64 --hidden_dim=256 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=10 \
--finetune_epochs=5 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--combine_method=logits --mi_type=conditional --finetune_logits=tilde \
--phase=train
done
