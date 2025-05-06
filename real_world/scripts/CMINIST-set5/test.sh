declare -a seed=(3)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=1028 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test \
--z_dim=32 --hidden_dim=128 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=3 \
--finetune_epochs=1 \
--decouple_alpha=1.0 --decouple_beta=10.0 --mmd_lambda=1.0 --domain_lambda=1.0 \
--combine_method=logits --mi_type=conditional --finetune_logits=tilde --loss_selection_mode=concat \
--phase=analysis
done
