declare -a seed=(8)
declare -a s_dim=(4)
declare -a lambda_vae=(5e-5)
declare -a device=(0)

for ((i=0;i<${#seed[@]};++i)); do
WANDB_MODE=online CUDA_VISIBLE_DEVICES=${device[0]} python ../../main2.py --root=../../../../da_datasets/pacs --batch-size=48 \
-d PACS -s A,S,P -t C -a resnet18 \
--name=PACS_test_To_C \
--z_dim=64 --s_dim=${s_dim[0]} \
--C_max=15 --beta=1 --lambda_vae=${lambda_vae[0]} --lambda_ent=0.1 \
-i=2500 \
--seed=${seed[i]} \
--vae_epochs=40 \
--train_batch_size=16 \
--unstable_epochs=20 \
--stable_epochs=20 \
--phase=train
done
