declare -a seed=(5)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=EnhancedColoredMNIST --batch-size=256 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test2 \
--z_dim=32 --hidden_dim=128 \
--lr=0.001 \
-i=100 \
--seed=${seed[i]} \
--train_epochs=5 \
--finetune_epochs=2 \
--decouple_alpha=1.0 --decouple_beta=2.0 --mmd_lambda=2.0 --domain_lambda=1.0 \
--combine_method=logits --mi_type=conditional --finetune_logits=tilde --loss_selection_mode=add \
--model_selection=OOD \
--phase=train --use_combined_inference
done
