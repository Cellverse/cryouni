#!/usr/bin/env bash
# =============================================================================
# CryoUNI Training Example
#
# Heterogeneous reconstruction with known consensus poses (e.g., from
# cryoSPARC or RELION). The model learns a physically grounded latent space
# where density reflects the Boltzmann distribution of conformational states.
#
# Prerequisites:
#   - Particle images in HDF5 format (see README for conversion instructions)
#   - Pretrained CryoUNI backbone weights (download from Zenodo)
#   - conda activate cryouni
# =============================================================================

# --- Paths (edit these) -------------------------------------------------------
PARTICLES=/path/to/particles.h5          # HDF5 particle images
PRETRAINED=/path/to/cryouni-b.ckpt       # Pretrained backbone weights
OUTPUT_DIR=output/my_experiment          # Training output directory
# ------------------------------------------------------------------------------

# --- Single-GPU training ------------------------------------------------------
python train.py \
    --config-file hetero_recon/configuration/hetero_only.yaml \
    --num-gpus 1 \
    DATAMODULE.DATASET.PARTICLE_PATH "$PARTICLES" \
    MODEL.BACKBONE.PRETRAINED_PATH "$PRETRAINED" \
    OUTPUT_DIR "$OUTPUT_DIR"


# --- Common config overrides --------------------------------------------------
# Override image size (spatial box size and Hartley transform size):
#   DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL 256
#   DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY 324
#
# Override batch size per GPU:
#   DATAMODULE.DATALOADER.TRAIN.BATCH_SIZE 8
#
# Override training epochs:
#   TRAINER.MAX_EPOCHS 50
#
# Use CryoUNI-S (small) backbone instead of base:
#   MODEL.BACKBONE.VIT_SCALE small
#
# Monitor training with TensorBoard:
#   tensorboard --logdir "$OUTPUT_DIR"


# --- Multi-GPU training (single node, 4 GPUs) ---------------------------------
# python train.py \
#     --config-file hetero_recon/configuration/hetero_only.yaml \
#     --num-nodes 1 \
#     --num-gpus 4 \
#     DATAMODULE.DATASET.PARTICLE_PATH "$PARTICLES" \
#     MODEL.BACKBONE.PRETRAINED_PATH "$PRETRAINED" \
#     OUTPUT_DIR "$OUTPUT_DIR"


# --- Multi-node SLURM training ------------------------------------------------
# Requires SLURM environment variables: SLURM_JOB_NUM_NODES, SLURM_NODEID,
# SLURM_JOB_NODELIST. Launch via: sbatch --nodes=2 train.sh
#
# export TIMM_FUSED_ATTN=1
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# torchrun \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --node_rank=$SLURM_NODEID \
#     --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
#     --master_port=12345 \
#     train.py \
#     --num-nodes $SLURM_JOB_NUM_NODES \
#     --num-gpus $NPROC_PER_NODE \
#     --config-file hetero_recon/configuration/hetero_only.yaml \
#     DATAMODULE.DATASET.PARTICLE_PATH "$PARTICLES" \
#     MODEL.BACKBONE.PRETRAINED_PATH "$PRETRAINED" \
#     OUTPUT_DIR "$OUTPUT_DIR"
