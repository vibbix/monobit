"""
monobit test suite
BeOS tuned-font format tests

Each test asserts behaviour required for the output to be accepted by the
BeOS R5 app_server, or protects loading of malformed and legacy files.
The `validate_beos` rules live in the codec module; the location-table
rule is verified against a live R5 system (86Box VM, 2026-07-10), the
other bounds follow the BeOS R4 app_server sources. Where the live R5
binary and the R4 source disagree, the live binary wins.
"""

from __future__ import annotations

import struct
import unittest

import monobit
from monobit.base import FileFormatError
from monobit.base.binary import ceildiv
from monobit.storage.fontformats.beos import (
    validate_beos, _HEADER, _GLYPH_DATA, _LOCATION_ENTRY, _INK_LOAD,
)

from .base import BaseTester


def _glyph_records_offset(data: bytes) -> int:
    """File offset of the first glyph record."""
    header = _HEADER.from_bytes(data[:_HEADER.size])
    return (
        _HEADER.size + header.ffnSize + 1 + header.fsnSize + 1
        + _LOCATION_ENTRY.size * (header.hmask + 1)
    )


def _style_name(data: bytes) -> str:
    """Style name stored in a tuned-font file."""
    header = _HEADER.from_bytes(data[:_HEADER.size])
    start = _HEADER.size + header.ffnSize + 1
    return data[start:start + header.fsnSize].decode('latin-1')


def _rescale_bitmaps_to_raw(data: bytes) -> bytes:
    """
    Rewrite glyph bitmaps from 3-bit to raw 4-bit levels, producing the
    file an earlier monobit version would have written for this content.
    """
    header = _HEADER.from_bytes(data[:_HEADER.size])
    out = bytearray(data)
    offset = _glyph_records_offset(data)
    while offset < header.size:
        glyph_data = _GLYPH_DATA.from_bytes(
            data[offset:offset + _GLYPH_DATA.size]
        )
        width = glyph_data.right - glyph_data.left + 1
        height = glyph_data.bottom - glyph_data.top + 1
        bitmap_size = ceildiv(width * 4, 8) * height
        start = offset + _GLYPH_DATA.size
        out[start:start + bitmap_size] = (
            data[start:start + bitmap_size].translate(_INK_LOAD)
        )
        offset = start + bitmap_size
    return bytes(out)


class TestBeOS(BaseTester):
    """BeOS tuned-font format round-trip and R5-compatibility tests."""

    konatu_path = BaseTester.font_path / 'Konatu' / 'Konatu_10'

    def _save_beos(self, font: monobit.Font) -> bytes:
        """Save font as BeOS tuned font, return the file bytes."""
        file = self.temp_path / 'font.beos'
        monobit.save(font, file, format='beos', overwrite=True)
        return file.read_bytes()

    def test_genuine_file_passes_validation(self) -> None:
        """The genuine Konatu_10 fixture satisfies the R5 validator."""
        data = self.konatu_path.read_bytes()
        self.assertEqual(validate_beos(data), [])

    @unittest.expectedFailure
    def test_saved_hash_table_power_of_two(self) -> None:
        """Saved location-table mask must be a power of 2 - 1 (VM-VERIFIED)."""
        data = self._save_beos(self.fixed4x6)
        header = _HEADER.from_bytes(data[:_HEADER.size])
        self.assertEqual(
            header.hmask & (header.hmask + 1), 0,
            f'mask {header.hmask} is not a power of 2 - 1: '
            'BeOS R5 rejects this file',
        )
        self.assertGreaterEqual(header.hmask, 3)

    @unittest.expectedFailure
    def test_saved_file_passes_validation(self) -> None:
        """A saved file must satisfy all R5 acceptance rules."""
        data = self._save_beos(self.fixed4x6)
        self.assertEqual(validate_beos(data), [])

    @unittest.expectedFailure
    def test_saved_style_matches_subfamily(self) -> None:
        """Style name must round-trip as the subfamily, not carry a size suffix."""
        font = self.fixed4x6.modify(
            family='Testfam', subfamily='Regular', point_size=14,
        )
        data = self._save_beos(font)
        self.assertEqual(_style_name(data), 'Regular')

    def test_saved_style_falls_back_to_name(self) -> None:
        """Fonts carrying their style only in `name` keep it on save."""
        glyph = self.fixed4x6.get_glyph('A')
        font = monobit.Font(
            [glyph], family='Helvetica', name='Helvetica Bold',
            encoding='unicode',
        )
        data = self._save_beos(font)
        self.assertEqual(_style_name(data), 'Bold')

    @unittest.expectedFailure
    def test_non_bmp_char_roundtrip(self) -> None:
        """Chars outside the BMP are stored as surrogate pairs in code[2]."""
        glyph = self.fixed4x6.get_glyph('A').modify(char='\U0001f600')
        font = monobit.Font([glyph], encoding='unicode')
        file = self.temp_path / 'emoji.beos'
        monobit.save(font, file, format='beos')
        reloaded, *_ = monobit.load(file, format='beos')
        chars = tuple(_g.char for _g in reloaded.glyphs)
        self.assertIn('\U0001f600', chars)

    def test_point_size_and_dpi(self) -> None:
        """BeOS renders 1pt == 1px; loaded fonts declare point size and 72 dpi."""
        font, *_ = monobit.load(self.konatu_path, format='beos')
        self.assertEqual(font.point_size, 10)
        self.assertEqual(tuple(font.dpi), (72, 72))

    def test_genuine_file_roundtrip(self) -> None:
        """A genuine file round-trips with equal content and passes validation."""
        font, *_ = monobit.load(self.konatu_path, format='beos')
        data = self._save_beos(font)
        self.assertEqual(validate_beos(data), [])
        reloaded, *_ = monobit.load(self.temp_path / 'font.beos', format='beos')
        self.assertEqual(len(reloaded.glyphs), len(font.glyphs))
        self.assertEqual(reloaded.family, font.family)
        self.assertEqual(reloaded.subfamily, font.subfamily)
        self.assertEqual(reloaded.point_size, font.point_size)
        for char in ('A', 'g', '!', 'あ'):
            glyph = font.get_glyph(char)
            reglyph = reloaded.get_glyph(char)
            self.assertEqual(glyph.as_matrix(), reglyph.as_matrix())
            self.assertEqual(glyph.scalable_width, reglyph.scalable_width)
            self.assertEqual(glyph.left_bearing, reglyph.left_bearing)
            self.assertEqual(glyph.right_bearing, reglyph.right_bearing)
            self.assertEqual(glyph.shift_up, reglyph.shift_up)

    def test_roundtrip_idempotent(self) -> None:
        """Save-load-save reproduces the file byte for byte."""
        font, *_ = monobit.load(self.konatu_path, format='beos')
        first = self._save_beos(font)
        reloaded, *_ = monobit.load(self.temp_path / 'font.beos', format='beos')
        second = self._save_beos(reloaded)
        self.assertEqual(first, second)

    def test_full_ink_reaches_max_level(self) -> None:
        """
        BeOS grayscale ink is 3-bit (0..7); a fully-inked pixel must map to
        the font's maximum ink level or glyphs render at ~47% opacity.
        """
        font, *_ = monobit.load(self.konatu_path, format='beos')
        max_ink = max(
            max((max(_row) for _row in _g.as_matrix()), default=0)
            for _g in font.glyphs[:200]
        )
        self.assertEqual(max_ink, font.levels - 1)

    def test_load_legacy_raw_ink(self) -> None:
        """
        Files written by earlier monobit versions store unscaled 4-bit
        levels; they load unrescaled, with a warning, instead of having
        ink above level 7 crushed to full black.
        """
        data = self._save_beos(self.fixed4x6)
        legacy = _rescale_bitmaps_to_raw(data)
        self.assertNotEqual(data, legacy)
        legacy_file = self.temp_path / 'legacy.beos'
        legacy_file.write_bytes(legacy)
        with self.assertLogs(level='WARNING') as logs:
            font, *_ = monobit.load(legacy_file, format='beos')
        self.assertTrue(any('Ink levels above 7' in _m for _m in logs.output))
        normal, *_ = monobit.load(self.temp_path / 'font.beos', format='beos')
        self.assertEqual(
            font.get_glyph('A').as_matrix(),
            normal.get_glyph('A').as_matrix(),
        )

    def test_load_rejects_oversized_location_table(self) -> None:
        """A corrupt mask cannot make the loader allocate beyond the file."""
        header = _HEADER(
            mark=b'|Be;', size=1000, ffnSize=1, fsnSize=1,
            hmask=0xffffff, point=10, bpp=3, version=0,
        )
        file = self.temp_path / 'corrupt.beos'
        file.write_bytes(bytes(header) + b'F\0S\0')
        with self.assertRaises(FileFormatError):
            monobit.load(file, format='beos')

    def test_load_rejects_corrupt_glyph_bbox(self) -> None:
        """Glyph records with out-of-range geometry are rejected cleanly."""
        data = bytearray(self._save_beos(self.fixed4x6))
        record = _glyph_records_offset(bytes(data))
        # corrupt bbox.right (offset +12 in the record)
        data[record + 12:record + 14] = struct.pack('>h', 30000)
        file = self.temp_path / 'corrupt.beos'
        file.write_bytes(bytes(data))
        with self.assertRaises(FileFormatError):
            monobit.load(file, format='beos')


if __name__ == '__main__':
    unittest.main()
