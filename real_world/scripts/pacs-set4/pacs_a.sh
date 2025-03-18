declare -a seed=(6)
declare -a device=(4)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled  CUDA_VISIBLE_DEVICES=${device[0]} python ../../main4.py --root=../../../../da_datasets/pacs --batch-size=48 \
-d PACS -s C,S,P -t A -a resnet18 \
--name=PACS_test_To_A \
--z_dim=64 \
-i=1000 \
--seed=${seed[i]} \
--train_batch_size=16 \
--train_epochs=20 \
--finetune_epochs=10 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train
done
