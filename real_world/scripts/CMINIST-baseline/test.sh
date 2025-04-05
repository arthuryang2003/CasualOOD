declare -a seed=(4)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../baseline.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=48 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test_To_3 \
--z_dim=32 --hidden_dim=128 \
-i=100 \
--seed=${seed[i]} \
--train_epochs=1 \
--finetune_epochs=1 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
