"""
monobit test suite
BeOS tuned-font format characterisation tests

Phase-0 red tests: each test asserts the behaviour required for the output
to be accepted by the BeOS R5 app_server. Tests marked expectedFailure
document known defects in the current codec; the commit that fixes a defect
removes the corresponding decorator.

Rule provenance:
- VM-VERIFIED: observed against the live BeOS R5 binary (86Box VM,
  2026-07-10). R5 rejects tuned files whose location-table mask is not a
  power of two minus one, and accepts + renders compliant files.
- SOURCE-DERIVED: read from the BeOS R4 app_server sources
  (servers/app/font_file.cpp get_tuned_info, font_set.cpp
  fc_read_char_from_file). Where the live R5 binary and the R4 source
  disagree, the live binary wins and these rules must be amended.
"""

from __future__ import annotations

import struct
import unittest

import monobit

from .base import BaseTester


def validate_beos_file(data: bytes) -> list[str]:
    """
    Validate a tuned-font file against BeOS R5 acceptance rules.

    Returns a list of problems; empty means app_server would accept the
    file (per the rules documented in the module docstring).
    """
    problems: list[str] = []
    if data[:4] != b'|Be;':
        return ['bad magic']
    if len(data) < 40:
        return ['file shorter than fixed header']
    total_length, ffn, fsn = struct.unpack('>IHH', data[4:12])
    rotation, shear = struct.unpack('>ff', data[12:20])
    hmask, = struct.unpack('>I', data[20:24])
    size, bpp, version = struct.unpack('>HBB', data[24:28])
    # SOURCE-DERIVED: name length limits (B_FONT_FAMILY/STYLE_LENGTH == 63)
    if ffn > 63 or fsn > 63:
        problems.append(f'name lengths out of range: {ffn}, {fsn}')
    # VM-VERIFIED: mask must be a power of two minus one (and >= 3);
    # a 211-glyph font saved with mask 419 is rejected as a bad file.
    if hmask & (hmask + 1) or hmask < 3:
        problems.append(f'location-table mask {hmask} not a power of 2 - 1')
    # SOURCE-DERIVED: point size bounds
    if not 1 < size <= 10000:
        problems.append(f'size {size} out of range')
    # SOURCE-DERIVED: bpp 1=B/W(RLE), 2=TV, 3=GRAY; version always 0
    if bpp not in (1, 2, 3):
        problems.append(f'invalid bpp {bpp}')
    if version != 0:
        problems.append(f'invalid version {version}')
    # SOURCE-DERIVED: names NUL-terminated, no embedded NULs
    names = data[36:36 + ffn + fsn + 2]
    if len(names) < ffn + fsn + 2:
        return problems + ['truncated names']
    family, fam_nul = names[:ffn], names[ffn]
    style, sty_nul = names[ffn + 1:ffn + 1 + fsn], names[ffn + 1 + fsn]
    if fam_nul or sty_nul or b'\0' in family or b'\0' in style:
        problems.append('names not correctly NUL-terminated')
    # location table
    loc_offset = 36 + ffn + fsn + 2
    if loc_offset + 8 * (hmask + 1) > len(data):
        return problems + ['location table does not fit in file']
    for i in range(hmask + 1):
        entry = data[loc_offset + 8 * i:loc_offset + 8 * i + 8]
        offset, = struct.unpack('>i', entry[:4])
        if offset <= 0:
            # empty slot (-1 in shipped files)
            continue
        problems.extend(_validate_glyph_record(data, offset))
        if len(problems) > 10:
            return problems + ['(further errors suppressed)']
    return problems


def _validate_glyph_record(data: bytes, offset: int) -> list[str]:
    """SOURCE-DERIVED per-glyph checks (fc_read_char_from_file)."""
    if offset + 24 > len(data):
        return [f'glyph record at {offset} does not fit in file']
    edge_left, edge_right = struct.unpack('>ff', data[offset:offset + 8])
    left, top, right, bottom = struct.unpack(
        '>hhhh', data[offset + 8:offset + 16]
    )
    escape_x, escape_y = struct.unpack(
        '>ff', data[offset + 16:offset + 24]
    )
    size_h = right - left + 1
    size_v = bottom - top + 1
    bitmap_size = ((size_h + 1) >> 1) * size_v
    problems: list[str] = []
    if size_h < 0 or size_v < 0:
        problems.append(f'glyph at {offset}: negative bitmap dimensions')
    if not 0 <= bitmap_size <= 128 * 1024:
        problems.append(f'glyph at {offset}: bitmap size {bitmap_size}')
    if left < -512 or right > 1024 or top < -1024 or bottom > 512:
        problems.append(f'glyph at {offset}: bbox out of range')
    # 1234567.0 in edge.left means 'edges not computed' and skips the check
    if edge_left != 1234567.0 and not (
            -2.0 <= edge_left <= 2.0 and -2.0 <= edge_right <= 2.0
    ):
        problems.append(f'glyph at {offset}: edges out of range')
    if not (-1000.0 <= escape_x <= 1000.0 and -1000.0 <= escape_y <= 1000.0):
        problems.append(f'glyph at {offset}: escapement out of range')
    if offset + 24 + max(bitmap_size, 0) > len(data):
        problems.append(f'glyph at {offset}: bitmap does not fit in file')
    return problems


class TestBeOS(BaseTester):
    """BeOS tuned-font format round-trip and R5-compatibility tests."""

    konatu_path = BaseTester.font_path / 'Konatu' / 'Konatu_10'

    def _save_beos(self, font: monobit.Font) -> bytes:
        """Save font as BeOS tuned font, return the file bytes."""
        file = self.temp_path / 'font.beos'
        monobit.save(font, file, format='beos')
        return file.read_bytes()

    def test_genuine_file_passes_validation(self) -> None:
        """The genuine Konatu_10 fixture satisfies the R5 validator."""
        data = self.konatu_path.read_bytes()
        self.assertEqual(validate_beos_file(data), [])

    @unittest.expectedFailure
    def test_saved_hash_table_power_of_two(self) -> None:
        """Saved location-table mask must be a power of 2 - 1 (VM-VERIFIED)."""
        data = self._save_beos(self.fixed4x6)
        hmask, = struct.unpack('>I', data[20:24])
        self.assertEqual(
            hmask & (hmask + 1), 0,
            f'mask {hmask} is not a power of 2 - 1: '
            'BeOS R5 rejects this file',
        )
        self.assertGreaterEqual(hmask, 3)

    @unittest.expectedFailure
    def test_saved_file_passes_validation(self) -> None:
        """A saved file must satisfy all R5 acceptance rules."""
        data = self._save_beos(self.fixed4x6)
        self.assertEqual(validate_beos_file(data), [])

    @unittest.expectedFailure
    def test_saved_style_matches_subfamily(self) -> None:
        """Style name must round-trip as the subfamily, not carry a size suffix."""
        font = self.fixed4x6.modify(
            family='Testfam', subfamily='Regular', point_size=14,
        )
        data = self._save_beos(font)
        ffn, fsn = struct.unpack('>HH', data[8:12])
        style = data[37 + ffn:37 + ffn + fsn].decode('latin-1')
        self.assertEqual(style, 'Regular')

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

    @unittest.expectedFailure
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


if __name__ == '__main__':
    unittest.main()
