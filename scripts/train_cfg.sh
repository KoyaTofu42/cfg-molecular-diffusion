#!/bin/bash

# Resume logic
DRIVE_CKPT="/content/drive/MyDrive/cfg_mol_diffusion/checkpoints/cfg_multi_property"
LOCAL_OUT="/content/project/e3_diffusion_for_molecules/outputs/cfg_multi_property"

if [ -d "$DRIVE_CKPT" ]; then
    mkdir -p /content/project/e3_diffusion_for_molecules/outputs/
    cp -r "$DRIVE_CKPT" "$LOCAL_OUT"
    echo "✅ Checkpoint restored from Drive"
    RESUME_FLAG="--resume outputs/cfg_multi_property"
else
    RESUME_FLAG=""
fi

python main_qm9.py \
    --n_epochs 3000 \
    --exp_name cfg_multi_property \
    --conditioning alpha mu gap \
    --cfg_dropout_prob 0.15 \
    --cfg_dropout_mode joint \
    --n_stability_samples 100 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 500 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --nf 128 \
    --n_layers 6 \
    --lr 2e-4 \
    --normalize_factors "[1,4,1]" \
    --test_epochs 20 \
    --ema_decay 0.9999 \
    --num_workers 2 \
    --save_model True \
    $RESUME_FLAG
