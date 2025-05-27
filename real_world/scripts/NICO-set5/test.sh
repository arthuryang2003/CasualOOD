declare -a seed=(4)
declare -a device=(7)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=NICO_Mixed --batch-size=16 \
-s train1,train2 -t test -a resnet18 \
--name=test \
--z_dim=64 --hidden_dim=256 \
--lr=0.0005 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=10 \
--finetune_epochs=33 \
--source_split_ratio=0.8 --target_split_ratio=0.2 \
--decouple_alpha=1.0 --decouple_beta=2.0 --mmd_lambda=2.0 --domain_lambda=1.0 \
--combine_method=logits --mi_type=conditional --finetune_logits=tilde --loss_selection_mode=add \
--model_selection=OOD \
--phase=train --use_combined_inference
done
