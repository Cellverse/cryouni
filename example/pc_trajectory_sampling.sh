# --- Paths (edit these) -------------------------------------------------------
RESULT=output/my_experiment             # Training output directory
IMAGES=/path/to/particles.h5            # HDF5 particle images
STAR=/path/to/particles.star            # RELION star file (for index_to_star)

CONFIG=$RESULT/csv_log/version_0/hparams.yaml
CKPT=$RESULT/regular_ckpts/last.ckpt
# ------------------------------------------------------------------------------


# =============================================================================
# PC Trajectory Sampling - Linear, Spiral, and Circle
#
# Generates a sequence of latent-space points along a PCA trajectory,
# optionally snaps them to the nearest data point (default: on data, output to on_data/;
# with --not-on-data uses off-manifold coordinates, output to off_data/), and produces volumes
# for each frame. All three trajectory types are demonstrated below.
# --volume-size: spatial dimension of output volumes (default: model's native size)
# =============================================================================

# 1. Linear trajectory along PC0 and PC1
python cli/pc_trajectory_sampling.py \
    --out "$RESULT/pc_trajectory/linear" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --traj-type linear \
    --pc-dim 0 \
    --num-samples 60 \
    --not-on-data

# 2. Spiral trajectory in the PC0-PC1 plane
python cli/pc_trajectory_sampling.py \
    --out "$RESULT/pc_trajectory/spiral" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --traj-type spiral \
    --pc-dim "0 1" \
    --num-samples 120 \
    --spiral-loops 3 \
    --spiral-scale 1.25 \
    --not-on-data

# 3. Circular trajectory in the PC0-PC1 plane
python cli/pc_trajectory_sampling.py \
    --out "$RESULT/pc_trajectory/circle" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --traj-type circle \
    --pc-dim "0 1" \
    --num-samples 60 \
    --not-on-data
