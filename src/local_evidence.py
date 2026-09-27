"""Offline multilingual evidence using the Mac's ICU transliteration rules.

No entity data, pretrained model or network lookup is used. This is a separate
feature schema: it never silently substitutes for AnyAscii in a saved model.
"""
import ctypes
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio, partial_ratio

from .disk_store import fetch_records
from .evidence import GENERIC


class LocalTransliterator:
    def __init__(self):
        self.lib = ctypes.CDLL('/usr/lib/libicucore.A.dylib')
        ushort = ctypes.POINTER(ctypes.c_uint16)
        integer = ctypes.POINTER(ctypes.c_int32)
        self.lib.utrans_openU.argtypes = [ushort, ctypes.c_int32, ctypes.c_int,
                                       ushort, ctypes.c_int32, ctypes.c_void_p, integer]
        self.lib.utrans_openU.restype = ctypes.c_void_p
        self.lib.utrans_transUChars.argtypes = [ctypes.c_void_p, ushort, integer,
                                              ctypes.c_int32, ctypes.c_int32, integer, integer]
        self.lib.utrans_close.argtypes = [ctypes.c_void_p]
        name = 'Any-Latin; Latin-ASCII; Lower'.encode('utf-16-le')
        buf = (ctypes.c_uint16 * (len(name) // 2)).from_buffer_copy(name)
        status = ctypes.c_int32(0)
        self.handle = self.lib.utrans_openU(buf, len(buf), 0, None, 0, None,
                                          ctypes.byref(status))
        if status.value > 0 or not self.handle:
            raise RuntimeError(f'Cannot initialize offline ICU transliterator: {status.value}')

    @lru_cache(maxsize=100000)
    def convert(self, value):
        if value.isascii():
            return value.lower()
        raw = value.encode('utf-16-le')
        n = len(raw) // 2
        capacity = max(256, n * 20 + 64)
        buf = (ctypes.c_uint16 * capacity)()
        ctypes.memmove(buf, raw, len(raw))
        length, limit, status = ctypes.c_int32(n), ctypes.c_int32(n), ctypes.c_int32(0)
        self.lib.utrans_transUChars(self.handle, buf, ctypes.byref(length), capacity,
                                   0, ctypes.byref(limit), ctypes.byref(status))
        if status.value > 0:
            raise RuntimeError(f'ICU transliteration failed: {status.value}')
        return ctypes.string_at(buf, length.value * 2).decode('utf-16-le')

    def close(self):
        self.convert.cache_clear()
        if self.handle:
            self.lib.utrans_close(self.handle)
            self.handle = None


def skeleton(value):
    for left, right in [('ph', 'f'), ('bh', 'b'), ('dh', 'd'), ('th', 't'),
                        ('kh', 'k'), ('sh', 's'), ('w', 'v')]:
        value = value.replace(left, right)
    return re.sub(r'(.)\1+', r'\1', re.sub('[aeiou]', '', value))


def sim(a, b, scorer=ratio):
    return scorer(a, b) / 100 if a and b else 0.


def records(frame, transliterator):
    result = {}
    for row in frame.itertuples(index=False):
        name = transliterator.convert(row.name_norm)
        address = transliterator.convert(row.address_norm)
        core = [v for v in name.split() if v not in GENERIC]
        numbers = [str(int(v)) for v in re.findall(r'\d+', address)]
        result[row.entity_id] = (name, address, core, numbers,
                                 ' '.join(map(skeleton, core)))
    return result


def features(pairs, con, transliterator):
    frame = pd.concat([fetch_records(con, 'anchors', pairs.anchor_rid.unique()),
                       fetch_records(con, 'targets', pairs.target_rid.unique())])
    text = records(frame, transliterator)
    output = []
    for pair in pairs.itertuples(index=False):
        n, a, core, nums, phon = text[pair.source1_entity_id]
        m, b, other, mnums, mphon = text[pair.candidate_entity_id]
        row = {}
        for prefix, left, right in [('name', n, m), ('core', ' '.join(core), ' '.join(other)),
                                     ('phonetic', phon, mphon), ('address', a, b)]:
            row[prefix + '_ratio'] = sim(left, right)
            row[prefix + '_sort'] = sim(left, right, token_sort_ratio)
            row[prefix + '_set'] = sim(left, right, token_set_ratio)
            row[prefix + '_partial'] = sim(left, right, partial_ratio)
        for prefix, left, right in [('name', core, other), ('address', a.split(), b.split())]:
            left = sorted(set(left), key=lambda t: (-len(t), t))[:16]
            right = sorted(set(right), key=lambda t: (-len(t), t))[:16]
            matrix = np.array([[ratio(x, y) / 100 for y in right] for x in left])
            for side, tokens, values in [
                ('left', left, matrix.max(axis=1) if left and right else np.zeros(len(left))),
                ('right', right, matrix.max(axis=0) if left and right else np.zeros(len(right)))]:
                row[prefix + '_' + side + '_soft'] = float(values.mean()) if len(values) else 0.
                row[prefix + '_' + side + '_weakest'] = float(values.min()) if len(values) else 0.
                row[prefix + '_' + side + '_unmatched'] = int((values < .65).sum())
                row[prefix + '_' + side + '_weighted'] = sum(len(t) * v for t, v in zip(tokens, values)) / max(1, sum(map(len, tokens)))
        row['number_sequence'] = sim(' '.join(nums), ' '.join(mnums))
        row['number_first_equal'] = int(bool(nums and mnums) and nums[0] == mnums[0])
        row['number_last_equal'] = int(bool(nums and mnums) and nums[-1] == mnums[-1])
        row['number_left_count'], row['number_right_count'] = len(nums), len(mnums)
        left, right = set(nums), set(mnums)
        row['number_jaccard'] = len(left & right) / max(1, len(left | right))
        row['number_subset'] = int(bool(left and right) and (left <= right or right <= left))
        row['number_left_only'], row['number_right_only'] = len(left - right), len(right - left)
        for key, value in [('left_name', n), ('right_name', m), ('left_address', a), ('right_address', b)]:
            row[key + '_length'] = len(value)
        row['source3'] = int(pair.candidate_entity_id.startswith('S3-'))
        output.append(row)
    return pd.DataFrame(output, index=pairs.index, dtype=np.float32).add_prefix('icu_')
