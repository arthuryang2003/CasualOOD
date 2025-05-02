declare -a seed=(4)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=1028 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test2 \
--z_dim=32 --hidden_dim=128 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=10 \
--finetune_epochs=2 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--combine_method=features --mi_type=conditional --finetune_logits=tilde \
--phase=train
done
