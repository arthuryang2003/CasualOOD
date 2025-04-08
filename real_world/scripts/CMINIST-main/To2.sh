declare -a seed=(4)
declare -a device=(6)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=256 \
-s '+90%','-90%' -t'+80%' -a lenet \
--name=group1 \
--z_dim=32 --hidden_dim=128 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
