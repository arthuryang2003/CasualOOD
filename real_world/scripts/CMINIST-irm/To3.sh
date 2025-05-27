declare -a seed=(4)
declare -a device=(7)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../IRM.py --data_dir=../../data --dataset=ColoredMNIST --batch-size=256 \
-s '+90%','+80%' -t'-90%' -a lenet \
--name=test \
--irm_lambda=100 --irm_anneal_iters=500 \
--lr=0.001 --wd=0 \
-i=100 \
--seed=${seed[i]} \
--train_epochs=50 \
--phase=train
done
