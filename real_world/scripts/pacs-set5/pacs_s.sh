declare -a seed=(8)
declare -a device=(5)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main5.py --root=../../../../da_datasets/pacs --batch-size=48 \
-d PACS -s C,P,A -t S -a resnet18 \
--name=PACS_test_To_S \
--z_dim=64 \
-i=1000 \
--seed=${seed[i]} \
--train_batch_size=16 \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
