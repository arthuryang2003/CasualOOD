declare -a seed=(8)
declare -a device=(3)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main3.py --data_dir=../../data --dataset=PACS --batch-size=48 \
-s A,C,S -t P -a resnet18 \
--name=group2 \
--z_dim=64 --hidden_dim=256 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done



