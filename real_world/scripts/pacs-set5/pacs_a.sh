declare -a seed=(7)
declare -a device=(7)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=PACS --batch-size=48 \
-s C,S,P -t A -a resnet18 \
--name=group2 \
--z_dim=64 --hidden_dim=256 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 --mmd_lambda=1.0 --domain_lambda=1.0 \
--combine_method=features --mi_type=cosine --finetune_logits=combined --loss_selection_mode=add \
--phase=train
done
