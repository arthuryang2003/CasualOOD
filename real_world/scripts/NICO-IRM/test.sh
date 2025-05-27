declare -a seed=(4)
declare -a device=(7)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../IRM.py --data_dir=../../data --dataset=NICO_Mixed --batch-size=16 \
-s train1,train2 -t test -a resnet18 \
--name=test \
--irm_lambda=10 --irm_anneal_iters=500 \
--lr=0.0005 \
-i=500 \
--seed=${seed[i]} \
--train_epochs=10 \
--source_split_ratio=0.8 --target_split_ratio=0.2 \
--phase=train
done
