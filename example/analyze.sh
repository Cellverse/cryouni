#!/usr/bin/env bash
# =============================================================================
# WAVE: Conformational Landscape Analysis Pipeline
#
# Runs the full WAVE (Watershed Analysis of Variational Embeddings) pipeline
# on a trained CryoUNI model. Steps:
#   1. Dimensionality reduction & GMM clustering
#   2. Peak detection across PC combinations
#   3. Watershed segmentation (energy-based basin assignment)
#   4. Trajectory planning (conformational transition paths)
#   5. (Optional) Export per-cluster STAR files for downstream refinement
#
# Prerequisites:
#   - Trained model checkpoint and config (from train.sh)
#   - Particle images in HDF5 format
#   - conda activate cryouni
# =============================================================================

# --- Paths (edit these) -------------------------------------------------------
RESULT=output/my_experiment             # Training output directory
IMAGES=/path/to/particles.h5            # HDF5 particle images
STAR=/path/to/particles.star            # RELION star file (for index_to_star)

CONFIG=$RESULT/csv_log/version_0/hparams.yaml
CKPT=$RESULT/regular_ckpts/last.ckpt
# ------------------------------------------------------------------------------

# =============================================================================
# Step 1: Dimensionality Reduction & GMM Clustering
#
# Encodes all particles to latent vectors, runs PCA + UMAP, fits a GMM with
# --cluster-num components, and generates volumes for each cluster center.
# Latent vectors are cached to all_z.pkl and reused by subsequent steps.
# =============================================================================
python cli/dim_reduction.py \
    --out "$RESULT/dim_reduction" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --cluster-num 10

# Export per-cluster STAR files (requires --star)
# python cli/index_to_star.py \
#     --out "$RESULT/dim_reduction" \
#     --star "$STAR"

# =============================================================================
# Step 2: Peak Detection
#
# Detects density peaks across all PC pair/triplet combinations using KDE.
# --max-pcs-2d: number of PCs for the 2D corner plot (C(n,2) pairs)
# --max-pcs-3d: number of PCs for 3D triplet analysis (C(n,3) triplets)
# --pcs-nd:     explicit PC indices for N-dimensional joint analysis
# =============================================================================
python cli/peak_detection.py \
    --out "$RESULT/peak_detection" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --max-pcs-2d 6 \
    --max-pcs-3d 4 \
    --pcs-nd "0 1 2 3"

# =============================================================================
# Step 3: Watershed Segmentation
#
# Density-based watershed clustering on selected principal components.
# --pc-dims:       space-separated PC indices to use (2D or 3D)
# --delta-delta-g: energy thresholds in kT for basin boundary tightening;
#                  "None" keeps all particles (no energy cutoff)
#
# Outputs per delta value:
#   watershed_labels.pkl, watershed_peaks.pkl  -- cluster assignments
#   watershed.png                              -- segmentation visualization
#   cluster_NNN.pkl                            -- per-cluster particle indices
# =============================================================================

# 2D watershed on PC0 and PC1
python cli/watershed.py \
    --out "$RESULT/watershed" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --pc-dims "0 1" \
    --delta-delta-g "1 2 3 4 None"

# 3D watershed on PC0, PC1, PC2
# python cli/watershed.py \
#     --out "$RESULT/watershed" \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --images "$IMAGES" \
#     --pc-dims "0 1 2" \
#     --delta-delta-g "1 2 3 4 None"

# Export per-cluster STAR files for each delta threshold
# for thresh in 1.0 2.0 3.0 4.0 none; do
#     python cli/index_to_star.py \
#         --out "$RESULT/watershed/PC0_PC1/delta_$thresh" \
#         --star "$STAR"
# done

# =============================================================================
# Step 4: Trajectory Planning
#
# Plans smooth conformational transition paths connecting density peaks.
# --method:     gradient (N-D), fmm (Fast Marching, 2D/3D only), linear (N-D)
# --num-steps:  number of frames along the trajectory
# --close/--open: closed loop (visits all peaks and returns) vs open path
#
# Outputs:
#   trajectory.npy, waypoints.npy          -- path coordinates
#   trajectory_summary.png                 -- visualization (2D/3D only)
#   energy_profile.png                     -- -log(density) along path
#   zero_padded/volume.*.mrc               -- volumes (zero-padded latent dims)
#   nearest_point/volume.*.mrc             -- volumes (nearest actual particle)
# =============================================================================

# 2D trajectory with Fast Marching Method (follows density ridges)
python cli/trajectory.py \
    --out "$RESULT/trajectory" \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --images "$IMAGES" \
    --pc-dims "0 1" \
    --method fmm \
    --num-steps 120

# 2D trajectory with gradient-driven refinement
# python cli/trajectory.py \
#     --out "$RESULT/trajectory" \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --images "$IMAGES" \
#     --pc-dims "0 1" \
#     --method gradient \
#     --num-steps 120

# 3D trajectory with Fast Marching Method
# python cli/trajectory.py \
#     --out "$RESULT/trajectory" \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --images "$IMAGES" \
#     --pc-dims "0 1 2" \
#     --method fmm \
#     --num-steps 120

# 3D trajectory with gradient-driven refinement
# python cli/trajectory.py \
#     --out "$RESULT/trajectory" \
#     --config "$CONFIG" \
#     --ckpt "$CKPT" \
#     --images "$IMAGES" \
#     --pc-dims "0 1 2" \
#     --method gradient \
#     --num-steps 120
