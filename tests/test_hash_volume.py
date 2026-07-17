from pathlib import Path
import unittest

import torch

from cellverse.math_utils import ht_object
from cellverse.math_utils.frequency import circularize
from cellverse.math_utils.grid import get_spatial_grid
from cellverse.math_utils.mask import get_radial_mask
from coach_pl.configuration import CfgNode

from hetero_recon.model.projector import RealSpaceRayProjector
from hetero_recon.model.volume.hash import ConditionalHashGridVolume, MultiResolutionHashEncoder


class MultiResolutionHashEncoderTest(unittest.TestCase):

    def test_dense_level_trilinear_interpolation(self) -> None:
        encoder = MultiResolutionHashEncoder(
            num_levels=1,
            features_per_level=1,
            log2_hashmap_size=8,
            base_resolution=2,
            max_resolution=2,
        )

        vertices = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(3))
        indices = encoder._linearize(vertices, resolution=2)
        values = vertices[:, 0] + 2 * vertices[:, 1] + 3 * vertices[:, 2]
        with torch.no_grad():
            encoder.tables[0][indices, 0] = values.float()

        coords = torch.tensor([[[-0.5, 0.25, 0.5]]], requires_grad=True)
        encoded = encoder(coords)
        scaled = (coords + 1.0) * 0.5 * 2
        expected = scaled[..., 0] + 2 * scaled[..., 1] + 3 * scaled[..., 2]

        torch.testing.assert_close(encoded[..., 0], expected)
        encoded.sum().backward()
        self.assertIsNotNone(coords.grad)
        self.assertTrue(torch.isfinite(coords.grad).all())

    def test_hashed_levels_receive_gradients(self) -> None:
        encoder = MultiResolutionHashEncoder(
            num_levels=2,
            features_per_level=2,
            log2_hashmap_size=4,
            base_resolution=2,
            max_resolution=8,
        )
        coords = torch.rand(2, 7, 3, requires_grad=True) * 2.0 - 1.0
        encoded = encoder(coords)

        self.assertEqual(encoded.shape, (2, 7, 4))
        encoded.square().sum().backward()
        self.assertTrue(all(table.grad is not None for table in encoder.tables))


class RealSpaceRayProjectorTest(unittest.TestCase):

    def test_radial_field_is_rotation_invariant(self) -> None:
        projector = RealSpaceRayProjector(spatial_image_size=8, ray_chunk_size=5)
        rotations = torch.stack([
            torch.eye(3),
            torch.tensor([
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        ])
        conformations = torch.zeros(2, 1)

        def radial_density(coords: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
            return torch.exp(-4.0 * coords.square().sum(dim=-1))

        projections = projector(radial_density, rotations, conformations)
        torch.testing.assert_close(projections[0], projections[1])

    def test_identity_projection_matches_hartley_central_slice(self) -> None:
        spatial_size = 8
        projector = RealSpaceRayProjector(spatial_image_size=spatial_size, ray_chunk_size=5)
        rotations = torch.eye(3).unsqueeze(0)
        conformations = torch.zeros(1, 1)

        def compact_density(coords: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
            return (1.0 - coords.square().sum(dim=-1)).clamp_min(0.0)

        projection_sp = projector(compact_density, rotations, conformations)[0]
        projection_ht = ht_object.ht2_center(circularize(projection_sp, last_n_dims=2))

        coords_3d = get_spatial_grid((spatial_size, spatial_size, spatial_size))
        volume_sp = compact_density(coords_3d.unsqueeze(0), conformations)[0]
        volume_ht = ht_object.ht3_center(circularize(volume_sp, last_n_dims=3))
        central_slice = volume_ht[volume_ht.shape[0] // 2]

        torch.testing.assert_close(projection_ht, central_slice, atol=1.0e-5, rtol=1.0e-5)


class ConditionalHashGridVolumeTest(unittest.TestCase):

    def test_forward_and_eval_volume_shapes_and_gradients(self) -> None:
        volume = ConditionalHashGridVolume(
            conformation_channels=3,
            hartley_image_size=8,
            num_levels=2,
            features_per_level=2,
            log2_hashmap_size=6,
            base_resolution=2,
            hidden_channels=8,
            depth=2,
            ray_chunk_size=8,
            checkpoint_rays=True,
            eval_chunk_size=64,
        )
        rotations = torch.eye(3).expand(2, -1, -1).clone()
        conformations = torch.randn(2, 3, requires_grad=True)

        prediction = volume(rotations, conformations)
        expected_points = get_radial_mask((9, 9), inner_radius=4).sum().item()
        self.assertEqual(prediction.shape, (2, expected_points))

        prediction.square().mean().backward()
        self.assertIsNotNone(conformations.grad)
        self.assertTrue(torch.isfinite(conformations.grad).all())
        self.assertGreater(conformations.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(volume.field.film.weight.grad)
        self.assertTrue(any(table.grad is not None for table in volume.field.encoder.tables))

        reconstructed = volume.eval_volume(torch.zeros(3), noise_std=1.0)
        self.assertEqual(reconstructed.shape, (8, 8, 8))
        sphere_mask = get_radial_mask((8, 8, 8), inner_radius=4)
        self.assertEqual(reconstructed[~sphere_mask].count_nonzero().item(), 0)

    def test_experimental_config_selects_hash_volume(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cfg = CfgNode.load_yaml_with_base(repo_root / "hetero_recon/configuration/hetero_hash.yaml")

        self.assertEqual(cfg.MODEL.VOLUME.NAME, "ConditionalHashGridVolume")
        self.assertEqual(cfg.MODEL.VOLUME.NUM_LEVELS, 16)
        self.assertEqual(cfg.MODEL.VOLUME.HIDDEN_CHANNELS, 64)


if __name__ == "__main__":
    unittest.main()
