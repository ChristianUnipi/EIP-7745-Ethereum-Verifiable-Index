"""
bloom_filter_ethereum.py
========================

The "logsBloom" that Ethereum uses TODAY: the structure that EIP-7745 wants
to replace, and therefore the baseline for eip7745_log_index.py on the same
dataset.

Exact formula (Yellow Paper, section 4.3.1, function M3:2048; go-ethereum,
core/types/bloom9.go)
-----------------------------------------------------------------------
One filter per block, 2048 bits (256 bytes). For every value v inserted (the
address of the contract that emitted the event, and the event signature
digest):

    h = hash(v)                                   # 32 bytes
    for i in {0, 2, 4}:                            # first 3 groups of 2 bytes
        b = big_endian_uint16(h[i:i+2]) & 0x7FF    # low 11 bits -> 0..2047
        set bit b of the filter

That is 3 bits per value (m=2048, k=3, fixed by the protocol: this module
only models the real Ethereum filter, no other shape). go-ethereum stores
the filter as a big-endian 2048-bit integer, so "bit b" is exactly bit b of
a Python int.

Hash note: as in the rest of the project, Ethereum's Keccak-256 is replaced
by SHA3-256 (same output size and statistical properties, different
padding). It does not change the false positive rate that is measured, but
the filters built here are not bit-identical to the mainnet ones.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

BLOOM_BIT_LENGTH = 2048
BLOOM_SIZE_BYTES = BLOOM_BIT_LENGTH // 8  # 256
BLOOM_K = 3
BIT_MASK_11 = 0x7FF  # 2**11 - 1: low 11 bits of each 2-byte group (2**11 = 2048)


def value_digest(value: bytes) -> bytes:
    """32-byte digest of a value (SHA3-256 as a stand-in for Keccak-256)."""
    return hashlib.sha3_256(value).digest()


def bit_positions(digest: bytes) -> List[int]:
    """The 3 bit positions (0..2047) that a value with this digest sets in the filter."""
    return [int.from_bytes(digest[i : i + 2], "big") & BIT_MASK_11 for i in (0, 2, 4)]


def bit_mask(digest: bytes) -> int:
    """The positions of `bit_positions` as an integer mask: a value may be
    in a filter iff `bits & mask == mask`."""
    mask = 0
    for position in bit_positions(digest):
        mask |= 1 << position
    return mask


class BlockBloom:
    """The bloom filter of one block, as one Python integer."""

    __slots__ = ("bits",)

    def __init__(self) -> None:
        self.bits = 0

    def add(self, value: bytes) -> None:
        self.bits |= bit_mask(value_digest(value))

    def might_contain(self, value: bytes) -> bool:
        mask = bit_mask(value_digest(value))
        return self.bits & mask == mask

    def bits_set(self) -> int:
        return bin(self.bits).count("1")

    def to_bytes(self) -> bytes:
        return self.bits.to_bytes(BLOOM_SIZE_BYTES, "big")


@dataclass
class BloomSearchResult:
    flagged_blocks: List[int]  # blocks whose filter says "maybe"
    confirmed_blocks: List[int]  # of those, blocks that really contain the value
    false_positive_blocks: List[int]  # flagged but not containing the value


class BloomIndex:
    """One bloom filter per block, as in the Ethereum block headers.

    Keeps, besides the filters, the exact set of values of every block: it is
    what a client gets when it re-reads the block's logs to confirm a match.
    Meant for small datasets and for tests; comparison.py streams the full
    dataset without keeping it.
    """

    def __init__(self) -> None:
        self.filters: Dict[int, BlockBloom] = {}
        self.values: Dict[int, Set[bytes]] = {}

    def add_event(self, block_id: int, address: bytes, topic: bytes) -> None:
        if block_id not in self.filters:
            self.filters[block_id] = BlockBloom()
            self.values[block_id] = set()
        for value in (address, topic):
            self.filters[block_id].add(value)
            self.values[block_id].add(value)

    def search(self, value: bytes) -> BloomSearchResult:
        flagged, confirmed, false_positive = [], [], []
        for block_id, block_filter in self.filters.items():
            if block_filter.might_contain(value):
                flagged.append(block_id)
                if value in self.values[block_id]:
                    confirmed.append(block_id)
                else:
                    false_positive.append(block_id)
        return BloomSearchResult(flagged, confirmed, false_positive)

    def size_bytes(self) -> int:
        """Bytes of all the filters: fixed per block, whatever it contains."""
        return len(self.filters) * BLOOM_SIZE_BYTES


def theoretical_fp_probability(distinct_values: float) -> float:
    """Standard bloom filter formula p = (1 - e^(-k n / m))^k for n distinct
    values, with the fixed Ethereum parameters m=2048, k=3."""
    return (1 - math.exp(-BLOOM_K * distinct_values / BLOOM_BIT_LENGTH)) ** BLOOM_K
