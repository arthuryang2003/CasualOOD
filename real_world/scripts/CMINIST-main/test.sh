declare -a seed=(4)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=48 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test \
--z_dim=32 --hidden_dim=128 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=2 \
--finetune_epochs=1 \
--decouple_alpha=1.0 --decouple_beta=10.0 --mmd_lambda=10.0 --domain_lambda=1.0 \
--combine_method=logits --mi_type=cosine --finetune_logits=tilde --loss_selection_mode=add \
--phase=train
done
