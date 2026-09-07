"""
monobit test suite
raster ink-level tests
"""

import unittest

from monobit.core import Glyph
from monobit.core.raster import Raster


class TestRaster(unittest.TestCase):
    """Test raster ink levels."""

    def test_rescale_ink(self):
        """Rescaling preserves each pixel's share of full ink."""
        raster = Raster.from_matrix([[0, 1], [7, 3]], inklevels=range(8))
        rescaled = raster.rescale_ink(16)
        self.assertEqual(rescaled.levels, 16)
        self.assertEqual(rescaled.as_matrix(), ((0, 2), (15, 6)))

    def test_rescale_ink_narrowing(self):
        """Rescaling to fewer levels collapses values evenly."""
        raster = Raster.from_matrix([list(range(16))], inklevels=range(16))
        rescaled = raster.rescale_ink(8)
        self.assertEqual(rescaled.levels, 8)
        self.assertEqual(
            rescaled.as_matrix()[0],
            (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
        )

    def test_bytes_with_fewer_ink_levels(self):
        """Values of a smaller scale keep their size in wider fields."""
        raster = Raster.from_bytes(
            b'\x07\x30', width=2, height=2, bits_per_pixel=4, ink_levels=8,
        )
        self.assertEqual(raster.levels, 8)
        self.assertEqual(raster.as_matrix(), ((0, 7), (3, 0)))
        self.assertEqual(
            raster.as_bytes(bits_per_pixel=4, ink_levels=8), b'\x07\x30'
        )

    def test_as_bytes_rejects_mismatched_ink_levels(self):
        """`ink_levels` must describe the raster it is packing."""
        raster = Raster.from_matrix([[0, 7]], inklevels=range(8))
        with self.assertRaises(ValueError):
            raster.as_bytes(bits_per_pixel=4, ink_levels=16)

    def test_glyph_passes_on_ink_levels(self):
        """Glyph forwards `ink_levels` to the raster rather than storing it."""
        glyph = Glyph.from_bytes(
            b'\x07\x30', width=2, height=2, bits_per_pixel=4, ink_levels=8,
        )
        self.assertEqual(glyph.levels, 8)
        self.assertEqual(glyph.rescale_ink(16).as_matrix(), ((0, 15), (6, 0)))


if __name__ == '__main__':
    unittest.main()
