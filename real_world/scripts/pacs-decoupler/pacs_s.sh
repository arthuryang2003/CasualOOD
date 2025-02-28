declare -a seed=(8)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=online CUDA_VISIBLE_DEVICES=${device[0]} python ../../main2.py --root=../../../../da_datasets/pacs --batch-size=48 \
-d PACS -s C,P,A -t S -a resnet18 \
--name=PACS_test_To_S \
--z_dim=64 \
-i=1000 \
--seed=${seed[i]} \
--train_epochs=10 \
--finetune_epochs=5 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
