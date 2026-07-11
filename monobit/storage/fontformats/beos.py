"""
monobit.storage.fontformats.beos - BeOS Bitmap Font

(c) 2024--2026 Rob Hagemans
licence: https://opensource.org/licenses/MIT
"""

import logging
from itertools import accumulate

from monobit.base.basetypes import FileFormatError, UnsupportedError
from monobit.base.binary import ceildiv
from monobit.base.struct import big_endian as be
from monobit.core import Font, Glyph
from monobit.storage import loaders, savers
from monobit.storage.streams import Stream
from monobit.storage.utils.limitations import ensure_levels, ensure_single

logger = logging.getLogger(__name__)

# http://www.eonet.ne.jp/~hirotsu/bin/bmf_format.txt

_HEADER = be.Struct(
    mark='4s',
    # total size of the file
    size='uint32',
    # > font-family-name size (not including the trailing null)
    ffnSize='uint16',
    # > font-style-name size (not including the trailing null)
    fsnSize='uint16',
    # rotation and shear angles in radians, 0.0 for upright
    rotation='float',
    shear='float',
    # location-table hash mask; must be a power of two minus one
    hmask='uint32',
    # > font-point (Bitmap fonts are enabled at this point number)
    point='uint16',
    # pixel format: 1 = B/W (RLE), 2 = TV scale, 3 = grayscale (4-bit packed)
    bpp='uint8',
    version='uint8',
    # uninitialised memory in fonts written by BeOS
    reserved='8s',
)

_FC_BLACK_AND_WHITE = 1
_FC_TV_SCALE = 2
_FC_GRAY_SCALE = 3

_LOCATION_ENTRY = be.Struct(
    offset='uint32',
    # utf-16: [char or high surrogate, low surrogate or 0]
    code_0='uint16',
    code_1='uint16',
)

_GLYPH_DATA = be.Struct(
    # ink edges of the scalable glyph, in em units;
    # 1234567.0 in edge_left means 'edges not computed'
    edge_left='float',
    edge_right='float',
    # bitmap bounding box relative to the baseline origin, y down
    left='int16',
    top='int16',
    right='int16',
    bottom='int16',
    # advance vector in (fractional) pixels
    x_escape='float',
    # `x_escape` is the LINEAR (outline) escapement stored in the file
    y_escape='float',
)

_EDGE_LEFT_NOT_COMPUTED = 1234567.0
_EDGE_RIGHT_NOT_COMPUTED = 1234568.0

_BEOS_MAGIC = b'|Be;'

def _char_from_codes(code_0: int, code_1: int) -> str:
    """Decode a location-entry utf-16 code unit pair to a character."""
    if 0xd800 <= code_0 < 0xdc00 and 0xdc00 <= code_1 < 0xe000:
        return chr(
            0x10000 + ((code_0 - 0xd800) << 10) + (code_1 - 0xdc00)
        )
    return chr(code_0)

def _location_table_size(n_glyphs: int) -> int:
    """
    the smallest power of two that holds the glyphs with 25% headroom, minimum 4. 
    BeOS rejects the file otherwise.
    """
    needed = ceildiv(n_glyphs * 5, 4)
    return max(4, 1 << (needed - 1).bit_length())

def _codes_from_char(char: str) -> tuple[int, int]:
    """Encode a character as a location-entry utf-16 code unit pair."""
    codepoint = ord(char)
    if codepoint > 0xffff:
        codepoint -= 0x10000
        return 0xd800 + (codepoint >> 10), 0xdc00 + (codepoint & 0x3ff)
    return codepoint, 0

def _location_hash(code_0: int, code_1: int, hmask: int) -> int:
    """Location-table hash function"""
    return (((code_0 << 3) ^ (code_0 >> 2)) + code_1) & hmask




# acceptance bounds enforced by the BeOS reader
# (font_file.cpp get_tuned_info, font_set.cpp fc_read_char_from_file)
_MAX_NAME_LENGTH = 63
_MAX_POINT_SIZE = 10000
_MAX_BITMAP_SIZE = 128 * 1024
_MIN_LEFT, _MAX_RIGHT = -512, 1024
_MIN_TOP, _MAX_BOTTOM = -1024, 512
_MAX_EDGE = 2.0
_MAX_ESCAPE = 1000.0


def _validate_header(header) -> list:
    """Check header fields against BeOS acceptance rules."""
    problems = []
    if header.mark != _BEOS_MAGIC:
        return ['bad magic']
    if header.ffnSize > _MAX_NAME_LENGTH or header.fsnSize > _MAX_NAME_LENGTH:
        problems.append(
            f'name lengths out of range: {header.ffnSize}, {header.fsnSize}'
        )
    if header.hmask & (header.hmask + 1) or header.hmask < 3:
        problems.append(
            f'location-table mask {header.hmask} not a power of 2 - 1'
        )
    if not 1 < header.point <= _MAX_POINT_SIZE:
        problems.append(f'point size {header.point} out of range')
    if header.bpp not in (
            _FC_BLACK_AND_WHITE, _FC_TV_SCALE, _FC_GRAY_SCALE
        ):
        problems.append(f'invalid bpp {header.bpp}')
    if header.version != 0:
        problems.append(f'invalid version {header.version}')
    return problems


def _validate_glyph_geometry(glyph_data) -> list:
    """Check glyph bitmap dimensions against BeOS acceptance rules."""
    width = glyph_data.right - glyph_data.left + 1
    height = glyph_data.bottom - glyph_data.top + 1
    bitmap_size = ((width + 1) >> 1) * height
    problems = []
    if width < 0 or height < 0:
        problems.append('negative bitmap dimensions')
    if not 0 <= bitmap_size <= _MAX_BITMAP_SIZE:
        problems.append(f'bitmap size {bitmap_size} out of range')
    if (
            glyph_data.left < _MIN_LEFT or glyph_data.right > _MAX_RIGHT
            or glyph_data.top < _MIN_TOP or glyph_data.bottom > _MAX_BOTTOM
        ):
        problems.append('bounding box out of range')
    return problems


def _validate_glyph_data(glyph_data) -> list:
    """Check a full glyph record against BeOS acceptance rules."""
    problems = _validate_glyph_geometry(glyph_data)
    # 1234567.0 in edge_left means 'edges not computed', skipping the check
    if glyph_data.edge_left != _EDGE_LEFT_NOT_COMPUTED and not (
            -_MAX_EDGE <= glyph_data.edge_left <= _MAX_EDGE
            and -_MAX_EDGE <= glyph_data.edge_right <= _MAX_EDGE
        ):
        problems.append('edges out of range')
    if not (
            -_MAX_ESCAPE <= glyph_data.x_escape <= _MAX_ESCAPE
            and -_MAX_ESCAPE <= glyph_data.y_escape <= _MAX_ESCAPE
        ):
        problems.append('escapement out of range')
    return problems


def validate_beos(data: bytes) -> list:
    """
    Check a tuned-font file against BeOS R5 acceptance rules.

    Returns a list of problems; empty means the BeOS app_server would
    accept the file. The location-table rule is verified against a live
    R5 system; the other bounds follow the BeOS R4 reader sources.
    """
    if len(data) < _HEADER.size:
        return ['file shorter than fixed header']
    header = _HEADER.from_bytes(data[:_HEADER.size])
    problems = _validate_header(header)
    if header.mark != _BEOS_MAGIC:
        return problems
    names_end = _HEADER.size + header.ffnSize + header.fsnSize + 2
    names = data[_HEADER.size:names_end]
    if len(names) < header.ffnSize + header.fsnSize + 2:
        return problems + ['truncated names']
    family, family_nul = names[:header.ffnSize], names[header.ffnSize]
    style = names[header.ffnSize + 1:header.ffnSize + 1 + header.fsnSize]
    style_nul = names[header.ffnSize + 1 + header.fsnSize]
    if family_nul or style_nul or b'\0' in family or b'\0' in style:
        problems.append('names not correctly NUL-terminated')
    table_size = _LOCATION_ENTRY.size * (header.hmask + 1)
    if names_end + table_size > len(data):
        return problems + ['location table does not fit in file']
    location_table = (_LOCATION_ENTRY * (header.hmask + 1)).from_bytes(
        data[names_end:names_end + table_size]
    )
    for entry in location_table:
        # the offset is read as signed by BeOS; empty slots hold -1
        if not 0 < entry.offset < 0x80000000:
            continue
        if entry.offset + _GLYPH_DATA.size > len(data):
            problems.append(
                f'glyph record at {entry.offset} does not fit in file'
            )
            continue
        glyph_data = _GLYPH_DATA.from_bytes(
            data[entry.offset:entry.offset + _GLYPH_DATA.size]
        )
        record_problems = _validate_glyph_data(glyph_data)
        width = glyph_data.right - glyph_data.left + 1
        height = glyph_data.bottom - glyph_data.top + 1
        if not record_problems and (
                entry.offset + _GLYPH_DATA.size
                + ceildiv(width * 4, 8) * height > len(data)
            ):
            record_problems.append('bitmap does not fit in file')
        problems.extend(
            f'glyph at {entry.offset}: {_p}' for _p in record_problems
        )
        if len(problems) > 10:
            return problems + ['(further errors suppressed)']
    return problems


@loaders.register(
    name='beos',
    magic=(_BEOS_MAGIC,)
)
def load_beos(instream: Stream, expand_ink: bool = True):
    """
    Load font from Be Bitmap Font file.

    expand_ink: rescale BeOS's 8 ink levels to the full 16-level range (default: True)
    """
    header = _HEADER.read_from(instream)
    if header.version != 0:
        raise FileFormatError( f'Unknown Be Bitmap Font version {header.version}.' )
    if header.bpp != _FC_GRAY_SCALE:
        raise UnsupportedError('Only grayscale Be Bitmap Fonts are supported.')
    if header.rotation != 0 or header.shear != 0:
        logger.warning('Nonzero rotation or shear angles are ignored.')
    # TODO: sanity check hmask
    familyName = instream.read(header.ffnSize+1)[:-1].decode('latin-1')
    styleName = instream.read(header.fsnSize+1)[:-1].decode('latin-1')
    logger.debug('family: %s', familyName)
    logger.debug('style: %s', styleName)

    table_size = _LOCATION_ENTRY.size * (header.hmask+1)
    table_bytes = instream.read(table_size)
    if len(table_bytes) != table_size:
        raise FileFormatError('Location table extends beyond end of file.')
    
    # hash table of pointers to glyphs, hashed by unicode codepoint
    location_table = (_LOCATION_ENTRY * (header.hmask+1)).from_bytes(table_bytes)
    location_dict = {
        _e.offset: _char_from_codes(_e.code_0, _e.code_1)
        for _e in location_table
        # the offset is read as signed by BeOS; empty slots hold -1
        if 0 < _e.offset < 0x80000000
    }

    glyphs = []
    while instream.tell() < header.size:
        pointer = instream.tell()
        glyph_data = _GLYPH_DATA.read_from(instream)
        problems = _validate_glyph_geometry(glyph_data)
        if problems:
            raise FileFormatError(
                f'Bad glyph record at offset {pointer}: '
                + '; '.join(problems)
            )
        # bitmap dimensions
        width = glyph_data.right - glyph_data.left + 1
        height = glyph_data.bottom - glyph_data.top + 1
        bitmap_size = ceildiv(width * 4, 8) * height
        glyph_bytes = instream.read(bitmap_size)
        if len(glyph_bytes) != bitmap_size:
            raise FileFormatError(
                f'Glyph bitmap at offset {pointer} extends beyond end of file.'
            )
        # TODO sanity check legacy_ink - older monobit didn't scale

        glyph = Glyph.from_bytes(
                        glyph_bytes, width=width, height=height, bits_per_pixel=4,
                        char=location_dict.get(pointer, None),
                        right_bearing=(int(glyph_data.x_escape + .5) - width - glyph_data.left),
                        left_bearing=glyph_data.left,
                        shift_up=-1-glyph_data.bottom,
                        scalable_width=glyph_data.x_escape,
                        ink_levels=8
                    )
        #useful for conversion, unneccesary for native
        if expand_ink:
            glyph = glyph.rescale_ink(16)
        glyphs.append(glyph)
    ## TODO: sanity check overhang

    # TODO: detect legacy_ink?
    return Font(
        glyphs,
        encoding='unicode',
        family=familyName,
        subfamily=styleName,
        point_size=header.point,
        # BeOS renders 1 point == 1 pixel, so the em is `point` pixels;
        # note that ascent + descent usually exceeds the em
        dpi=72,
    )


@savers.register(linked=load_beos)
def save_beos(fonts, outstream):
    """Save font to BeOS file."""
    font = ensure_single(fonts)
    if font.levels > 8:
        logger.warning(
            'BeOS stores 8 ink levels; %d-level ink will be quantised.',
            font.levels,
        )
    # 4 bits per pixel
    font = ensure_levels(font, 16)
    font = font.label()
    # drop multi-codepoint sequences and unlabelled glyphs
    glyphs = tuple(_g for _g in font.glyphs if _g.char and len(_g.char) == 1)
    # create header
    style_name = font.subfamily or font.name[len(font.family):].strip()
    family_name = font.family
    if (
            len(family_name) > _MAX_NAME_LENGTH
            or len(style_name) > _MAX_NAME_LENGTH
        ):
        logger.warning(
            'Family and style names longer than %d are truncated.',
            _MAX_NAME_LENGTH,
        )
        family_name = family_name[:_MAX_NAME_LENGTH]
        style_name = style_name[:_MAX_NAME_LENGTH]
    count = _location_table_size(len(glyphs))
    header = _HEADER(
        mark=_BEOS_MAGIC,
        # size='uint32',
        ffnSize=len(family_name),
        fsnSize=len(style_name),
        hmask=count-1,
        point=font.point_size,
        bpp=_FC_GRAY_SCALE,
        version=0,
    )
    # create glyph table
    glyph_data = tuple(
        bytes(_GLYPH_DATA(
            edge_left=_EDGE_LEFT_NOT_COMPUTED,
            edge_right=_EDGE_RIGHT_NOT_COMPUTED,
            left=_g.left_bearing,
            top=(-1-_g.shift_up) -_g.height + 1,
            right=_g.width + _g.left_bearing - 1,
            bottom=-1-_g.shift_up,
            x_escape=_g.scalable_width,
            # y_escape='float',
        ))
        for _g in glyphs
    )
    strike_offset = (
        _HEADER.size + header.ffnSize + 1 + header.fsnSize + 1
        + _LOCATION_ENTRY.size * count
    )
    glyph_bytes = tuple(_g.rescale_ink(8).as_bytes(bits_per_pixel=4, ink_levels=8) for _g in glyphs)
    offsets = accumulate(
        (len(_g) + len(_s) for _g, _s in zip(glyph_data, glyph_bytes)),
        initial=strike_offset,
    )
    # create location entries
    codes = tuple(_codes_from_char(_g.char) for _g in glyphs)
    loc_entries = tuple(
        _LOCATION_ENTRY(offset=_offs, code_0=_c0, code_1=_c1)
        for (_c0, _c1), _offs in zip(codes, offsets)
    )
    strike = b''.join(
        b''.join((_data, _bytes))
        for _data, _bytes in zip(glyph_data, glyph_bytes)
    )
    header.size = strike_offset + len(strike)
    # create hash table
    hashes = (
        _location_hash(_c0, _c1, header.hmask)
        for _c0, _c1 in codes
    )
    location_table = [None] * count
    for entry, hash in zip(loc_entries, hashes):
        while location_table[hash] is not None:
            hash = (hash + 1) & header.hmask
        location_table[hash] = entry
    location_table = (_LOCATION_ENTRY * count)(*(
        _entry if _entry else _LOCATION_ENTRY(offset=0xffffffff)
        for _entry in location_table
    ))
    outstream.write(bytes(header))
    outstream.write(family_name.encode('latin-1', 'replace')+ b'\0')
    outstream.write(style_name.encode('latin-1', 'replace')+ b'\0')
    outstream.write(bytes(location_table))
    outstream.write(bytes(strike))
