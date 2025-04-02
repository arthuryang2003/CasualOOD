declare -a seed=(4)
declare -a s_dim=(64)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main.py --data_dir=../../data --dataset=PACS --batch-size=48 \
-s A,S,P -t C -a resnet18 \
--name=group1 \
--z_dim=64 --hidden_dim=256 \
-i=100 \
--seed=${seed[i]} \
--train_epochs=1 \
--finetune_epochs=1 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
