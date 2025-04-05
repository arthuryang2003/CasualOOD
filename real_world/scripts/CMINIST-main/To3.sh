declare -a seed=(5)
declare -a device=(1)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=256 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=group3 \
--z_dim=32 --hidden_dim=128 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
