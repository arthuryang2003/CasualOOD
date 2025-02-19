declare -a seed=(8)
declare -a s_dim=(4)
declare -a lambda_vae=(5e-5)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${device[0]} python ../../main2.py --root=../../../../da_datasets/pacs --batch-size=48 \
-d PACS -s C,S,P -t A -a resnet18 \
--name=PACS_test_To_A \
--z_dim=128 --s_dim=64 \
-i=1000 \
--seed=${seed[i]} \
--decoupler_epochs=40 \
--train_batch_size=16 \
--unstable_epochs=40 \
--stable_epochs=40 \
--decouple_alpha=1.0 --decouple_beta=10.0 \
--phase=train


done
