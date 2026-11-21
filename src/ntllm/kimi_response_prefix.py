"""Assemble the Kimi-K3 XTML response prefix in the runner's NTLK format."""

import struct


# Preserve the boundaries of Moonshot's build_chat_segments. In particular,
# combining the tag name and its attributes could change their BPE token IDs.
RESPONSE_SEGMENTS = (
    '<|open|>', 'message', ' role', '="', 'user', '"', '<|sep|>',
    'Write one paragraph.',
    '<|close|>', 'message', '<|sep|>', '<|end_of_msg|>',
    '<|open|>', 'message', ' role', '="', 'assistant', '"', '<|sep|>',
    '<|open|>', 'think', '<|sep|>', '<|close|>', 'think', '<|sep|>',
    '<|open|>', 'response', '<|sep|>',
)
MARKER_IDS = {'<|open|>': 163587, '<|close|>': 163588, '<|sep|>': 163589, '<|end_of_msg|>': 163586}
MARKER_PROBE = ''.join(MARKER_IDS)
ORDINARY_SEGMENTS = tuple(dict.fromkeys(part for part in RESPONSE_SEGMENTS if part not in MARKER_IDS))


def read_token_rows(path):
    """Return the separately encoded prefix and body IDs for each NTLK row."""
    with path.open('rb') as handle:
        magic, count = struct.unpack('<4sI', handle.read(8))
        if magic != b'NTLK':
            raise ValueError(f'Expected an NTLK token file at {path}')
        rows = []
        for _ in range(count):
            length, prefix_length = struct.unpack('<II', handle.read(8))
            if not 0 < prefix_length <= length:
                raise ValueError(f'Invalid NTLK prefix length in {path}')
            ids = struct.unpack(f'<{length}i', handle.read(4 * length))
            rows.append((ids[:prefix_length], ids[prefix_length:]))
        if handle.read(1):
            raise ValueError(f'Unexpected trailing data in {path}')
    return rows


def write_response_tokens(baseline, probe, output, count):
    """Replace the baseline prefix after verifying the mounted GGUF XTML IDs.

    The probe tokenizes ORDINARY_SEGMENTS as the bodies and MARKER_PROBE as
    the common prefix in a single ntl-segrun tokenize invocation.
    """
    pieces = {marker: (token,) for marker, token in MARKER_IDS.items()}
    expected_markers = tuple(MARKER_IDS.values())
    for part, (markers, ids) in zip(ORDINARY_SEGMENTS, read_token_rows(probe), strict=True):
        if markers != expected_markers:
            raise ValueError(f'Kimi-K3 GGUF XTML markers are {markers}, expected {expected_markers}')
        if not ids:
            raise ValueError(f'Empty tokenization for the Kimi-K3 XTML fragment {part!r}')
        pieces[part] = ids
    prefix = tuple(token for part in RESPONSE_SEGMENTS for token in pieces[part])

    rows = read_token_rows(baseline)
    if not rows or len(rows) != count:
        raise ValueError('Kimi-K3 baseline token rows do not match the selected passages')
    baseline_prefix = rows[0][0]
    if any(old != baseline_prefix or not body for old, body in rows):
        raise ValueError('Kimi-K3 baseline NTLK requires a shared prefix and a nonempty body in every row')

    with output.open('wb') as handle:
        handle.write(struct.pack('<4sI', b'NTLK', count))
        for _, body in rows:
            length = len(prefix) + len(body)
            handle.write(struct.pack('<II', length, len(prefix)))
            handle.write(struct.pack(f'<{length}i', *prefix, *body))
    return {
        'prefix_mode': 'xtml_user_empty_think_response',
        'prefix': ''.join(RESPONSE_SEGMENTS),
        'prefix_tokens': len(prefix),
        'baseline_prefix_tokens': len(baseline_prefix),
    }
