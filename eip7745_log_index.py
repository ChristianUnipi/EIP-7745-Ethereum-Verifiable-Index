"""
eip7745_log_index.py
====================

Implementation of the "Log Index" data structure of EIP-7745
(https://eips.ethereum.org/EIPS/eip-7745), adapted to the "events" dataset:
Ethereum blocks 14,000,000-14,999,999, one record per *unique* event
occurrence in a block (20-byte contract address + 32-byte event signature
digest).

--------------------------------------------------------------------------
* Every event produces TWO insertions into the filter maps, exactly like a
  real log: one for the address and one for the topic (here the event
  signature digest, i.e. topics[0]). Each insertion takes the next global
  sequential position (`next_entry`).
* Positions are grouped into *filter maps* of VALUES_PER_MAP positions.
* Inside a map, a value is assigned to a *row* by a hash of
  (value, masked map index, layer). The map index is "masked" with the
  mapping frequency of the layer (`map - map % frequency`), so the row of a
  value stays the same for a whole group of consecutive maps.
* If the row selected at a layer already holds the maximum number of
  columns for that layer, the value moves on to the next layer (lower
  mapping frequency, larger capacity). There are 4 layers.
* What is stored in a row is a *column*: the in-map position of the entry
  in the high bits (in the clear) and a hash-derived tag in the low
  `LOG2_MAP_WIDTH - LOG2_VALUES_PER_MAP` bits.
* Search: for every map, compute the value's row at layer 0, read it, and
  move on to the next layer ONLY if that row is full (otherwise the value
  can never have overflowed, because insertion only leaves a layer when its
  row is full). Each stored column gives a candidate position (high bits);
  the tag filters most wrong candidates; the survivors are confirmed
  against the real entry stored at that position. A survivor that fails the
  confirmation is a *false candidate* (false positive).

What is NOT implemented, and why
--------------------------------
No node-by-node binary Merkle tree is built: 
with hundreds of millions of entries it is intractable in pure
Python. The sizes and the proof sizes are computed analytically from the
tree geometry (fully determined by the parameters) and from the counts that
are really observed (see the "Size and proof formulas" section).

Memory model
------------
Only the rows of the *current* map are kept in memory (`array('I')` per
row, 4 bytes per column). When a map is completed an optional callback
(`on_map_complete`) can inspect it (this is how comparison.py measures
false positives on every map of the full dataset), after which it is
discarded, unless `retain_maps=True` (used by `search()` and by the tests).
"""

import hashlib
import math
from array import array
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

HASH_SIZE = 32  # bytes of a hash / Merkle root (the exact hash does not matter)

# Rows are cached per layer for the current masked-map group (the same value
# is inserted over and over, and its row does not change inside a group).
_ROW_CACHE_LIMIT = 200_000


def _hash64(data: bytes) -> int:
    """64-bit hash used for rows and column tags (BLAKE2s, from the stdlib).

    The real EIP uses SHA-256 for rows and FNV-1a for columns; the mechanism
    only needs a well distributed hash, so a faster one is used here."""
    return int.from_bytes(hashlib.blake2s(data, digest_size=8).digest(), "big")


def value_fingerprint(value: bytes) -> int:
    """64-bit fingerprint of a value, used to *confirm* a candidate.

    It stands in for "read the real entry at that position and compare":
    keeping the full 20/32-byte values of a 16M-position map would need too
    much memory. XOR of three 8-byte windows, so that vanity addresses with
    a long common prefix or suffix do not collide. The tests cross-check the
    result against exact ground truth (see `tracked_values`)."""
    return (
        int.from_bytes(value[:8], "big")
        ^ int.from_bytes(value[8:16], "big")
        ^ int.from_bytes(value[-8:], "big")
    )


@dataclass(frozen=True)
class IndexParameters:
    """All the parameters of the structure, with the names and the default
    values of the real EIP (log_index.py):
    LOG2_VALUES_PER_MAP=16, LOG2_MAP_HEIGHT=16, LOG2_MAP_WIDTH=24,
    LOG2_MAPS_PER_EPOCH=10, LOG2_EPOCH_HISTORY=24,
    LOG2_MAPPING_FREQUENCY=[10, 6, 2, 0], MAX_ROW_LENGTH=[8, 168, 2728, 10920].
    """

    log2_values_per_map: int = 16
    log2_map_height: int = 16
    log2_map_width: int = 24
    log2_maps_per_epoch: int = 10
    log2_epoch_history: int = 24
    mapping_frequency_by_layer: Tuple[int, ...] = (1024, 64, 4, 1)
    max_row_length_by_layer: Tuple[int, ...] = (8, 168, 2728, 10920)

    def __post_init__(self) -> None:
        if self.log2_map_width < self.log2_values_per_map:
            raise ValueError("LOG2_MAP_WIDTH must be >= LOG2_VALUES_PER_MAP (the position must fit in the column)")
        if self.log2_map_width > 32:
            raise ValueError("LOG2_MAP_WIDTH > 32 is not supported (columns are stored as uint32)")
        if len(self.mapping_frequency_by_layer) != len(self.max_row_length_by_layer):
            raise ValueError("one mapping frequency and one maximum row length are needed per layer")

    @property
    def values_per_map(self) -> int:
        return 1 << self.log2_values_per_map

    @property
    def map_height(self) -> int:
        return 1 << self.log2_map_height

    @property
    def maps_per_epoch(self) -> int:
        return 1 << self.log2_maps_per_epoch

    @property
    def tag_bits(self) -> int:
        """Bits of the column left for the hash tag: LOG2_MAP_WIDTH -
        LOG2_VALUES_PER_MAP. With VALUES_PER_MAP = 2**24 and the default
        LOG2_MAP_WIDTH = 24 this is 0: the column is only the position."""
        return self.log2_map_width - self.log2_values_per_map

    @property
    def bytes_per_column(self) -> int:
        """3 bytes with LOG2_MAP_WIDTH=24 (log_index_proof_format.md: "3 bytes per list entry")."""
        return (self.log2_map_width + 7) // 8

    @property
    def num_layers(self) -> int:
        return len(self.mapping_frequency_by_layer)


class MapState:
    """The content of one filter map."""

    __slots__ = ("index", "rows", "fingerprints", "filled", "tracked_counts")

    def __init__(self, index: int, fingerprints: array) -> None:
        self.index = index
        # (layer << 32 | row) -> columns stored in that row
        self.rows: Dict[int, array] = {}
        # in-map position -> fingerprint of the value stored there
        self.fingerprints = fingerprints
        self.filled = 0
        # exact ground truth for the tracked values: query id -> occurrences in this map
        self.tracked_counts: Dict[int, int] = {}


@dataclass
class SearchResult:
    positions: List[int]  # confirmed global positions of the value
    columns_read: int  # columns read from the rows
    false_candidates: int  # tag matched but the entry is a different value
    layers_consulted: int  # rows read (one per layer consulted, per map)


class LogIndex:
    """The EIP-7745 Log Index.

    on_map_complete(index, map_state): called when a map is complete (and by
        `finish()` for the last one).
    retain_maps: keep every completed map (needed by `search()`; memory grows).
    tracked_values: {value: id}; the index counts the exact number of
        occurrences of these values in every map (ground truth used to check
        the recall of the structure).
    """

    def __init__(
        self,
        parameters: Optional[IndexParameters] = None,
        retain_maps: bool = False,
        on_map_complete: Optional[Callable[["LogIndex", MapState], None]] = None,
        tracked_values: Optional[Dict[bytes, int]] = None,
    ) -> None:
        self.p = parameters or IndexParameters()
        self.retain_maps = retain_maps
        self.on_map_complete = on_map_complete
        self.tracked_values = tracked_values

        self._tag_bits = self.p.tag_bits
        self._tag_mask = (1 << self._tag_bits) - 1
        self._height_mask = self.p.map_height - 1
        self._layer_bytes = [bytes([layer]) for layer in range(self.p.num_layers)]

        self.next_entry = 0
        self.num_maps = 0
        self.maps: List[MapState] = []
        self._current: Optional[MapState] = None
        self._shared_fingerprints = None if retain_maps else array("Q", bytes(8 * self.p.values_per_map))

        # Aggregate statistics
        self.columns_per_layer: List[int] = [0] * self.p.num_layers
        # Entries that found every layer full and were stored in the last one
        # anyway, beyond its nominal capacity (see insert_value).
        self.entries_beyond_last_layer = 0

        self._row_cache: List[Dict[bytes, int]] = [{} for _ in range(self.p.num_layers)]
        self._row_cache_group: List[int] = [-1] * self.p.num_layers

    # ------------------------------------------------------------------
    # Rows and columns
    # ------------------------------------------------------------------
    def row_index_for_masked_map(self, value: bytes, masked_map: int, layer: int) -> int:
        """Row of `value` given an ALREADY masked map index."""
        data = value + masked_map.to_bytes(8, "big") + self._layer_bytes[layer]
        return _hash64(data) & self._height_mask

    def row_index(self, value: bytes, map_index: int, layer: int) -> int:
        """Row in which `value` is looked for / stored, in map `map_index`, at `layer`.
        Same role as get_row_index in log_index.py (masked_map_index)."""
        frequency = self.p.mapping_frequency_by_layer[layer]
        return self.row_index_for_masked_map(value, map_index - (map_index % frequency), layer)

    def column(self, value: bytes, position: int, position_in_map: int) -> int:
        """Column of `value` if it sits at global `position`: the position
        inside the map in the clear in the high bits, a hash tag of (value,
        global position) in the low bits (same layout as get_column_index)."""
        if self._tag_bits == 0:
            return position_in_map
        tag = _hash64(value + position.to_bytes(8, "big")) & self._tag_mask
        return (position_in_map << self._tag_bits) | tag

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------
    def _complete_current_map(self) -> None:
        current = self._current
        if current is None:
            return
        if self.on_map_complete is not None:
            self.on_map_complete(self, current)
        if self.retain_maps:
            self.maps.append(current)

    def _start_map(self, map_index: int) -> None:
        self._complete_current_map()
        fingerprints = (
            array("Q", bytes(8 * self.p.values_per_map)) if self.retain_maps else self._shared_fingerprints
        )
        self._current = MapState(map_index, fingerprints)
        self.num_maps += 1

    def finish(self) -> None:
        """Complete the last (possibly partial) map."""
        self._complete_current_map()
        self._current = None

    def insert_value(self, value: bytes) -> int:
        """Insert one value (address or topic); returns its global position."""
        position = self.next_entry
        self.next_entry += 1
        map_index = position // self.p.values_per_map
        if self._current is None or map_index != self._current.index:
            self._start_map(map_index)
        current = self._current
        position_in_map = position - map_index * self.p.values_per_map

        current.fingerprints[position_in_map] = value_fingerprint(value)
        current.filled = position_in_map + 1
        column = self.column(value, position, position_in_map)

        rows = current.rows
        num_layers = self.p.num_layers
        placed = False
        for layer in range(num_layers):
            frequency = self.p.mapping_frequency_by_layer[layer]
            masked_map = map_index - (map_index % frequency)
            cache = self._row_cache[layer]
            if masked_map != self._row_cache_group[layer]:
                cache.clear()
                self._row_cache_group[layer] = masked_map
            row = cache.get(value)
            if row is None:
                row = self.row_index_for_masked_map(value, masked_map, layer)
                if len(cache) >= _ROW_CACHE_LIMIT:
                    cache.clear()
                cache[value] = row
            key = (layer << 32) | row
            row_columns = rows.get(key)
            if row_columns is None:
                row_columns = rows[key] = array("I")
            if len(row_columns) < self.p.max_row_length_by_layer[layer]:
                row_columns.append(column)
                self.columns_per_layer[layer] += 1
                placed = True
                break

        if not placed:
            # Every layer is full. With extremely recurrent values (e.g. the
            # `Transfer` event signature, shared by thousands of ERC-20
            # contracts) this can happen even with the real EIP capacities.
            # Instead of losing the entry, or aborting the run, it is stored
            # in the last layer beyond its nominal capacity, and counted.
            row_columns.append(column)  # `row_columns` is the last layer's row
            self.columns_per_layer[num_layers - 1] += 1
            self.entries_beyond_last_layer += 1

        if self.tracked_values is not None:
            query_id = self.tracked_values.get(value)
            if query_id is not None:
                current.tracked_counts[query_id] = current.tracked_counts.get(query_id, 0) + 1
        return position

    def insert_event(self, address: bytes, topic: bytes) -> Tuple[int, int]:
        """Insert an event occurrence: two consecutive entries, as
        log_index_add_log_entries does for a log with a single topic."""
        return self.insert_value(address), self.insert_value(topic)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def probe_map(
        self,
        map_state: MapState,
        value: bytes,
        fingerprint: int,
        row_provider: Optional[Callable[[int], int]] = None,
        collect_positions: bool = False,
    ) -> Tuple[int, int, int, int, Optional[List[int]]]:
        """Look for `value` in ONE map.

        Returns (columns_read, true_matches, false_candidates,
        layers_consulted, positions).

        Layers are consulted in order and the loop stops at the first row
        that is not full: insertion only moves to the next layer when the
        row is full, so an entry can only be in a later layer if every
        earlier row of that value is full.

        A column gives a candidate position (its high bits). If the entry
        stored at that position is the searched value it is a true match;
        otherwise it is a false candidate when the column tag equals the tag
        the searched value would have at that position (with 0 tag bits every
        column is a candidate). This is what a client would fetch and then
        discard, so it is counted as a false positive. (A true match always
        passes the tag test, so checking the fingerprint first only saves
        hashing: the counts are identical.)

        `row_provider(layer)` lets a caller reuse already computed rows.
        """
        tag_bits = self._tag_bits
        max_lengths = self.p.max_row_length_by_layer
        fingerprints = map_state.fingerprints
        base = map_state.index * self.p.values_per_map
        columns_read = true_matches = false_candidates = layers_consulted = 0
        positions: Optional[List[int]] = [] if collect_positions else None

        for layer in range(self.p.num_layers):
            row = row_provider(layer) if row_provider is not None else self.row_index(value, map_state.index, layer)
            layers_consulted += 1
            row_columns = map_state.rows.get((layer << 32) | row)
            if row_columns is None:
                break
            for stored_column in row_columns:
                columns_read += 1
                position_in_map = stored_column >> tag_bits
                if fingerprints[position_in_map] == fingerprint:
                    true_matches += 1
                    if positions is not None:
                        positions.append(base + position_in_map)
                elif tag_bits == 0 or stored_column == self.column(value, base + position_in_map, position_in_map):
                    false_candidates += 1
            if len(row_columns) < max_lengths[layer]:
                break
        return columns_read, true_matches, false_candidates, layers_consulted, positions

    def search(self, value: bytes) -> SearchResult:
        """Search a value in every retained map (`retain_maps=True`)."""
        if not self.retain_maps:
            raise RuntimeError("search() needs retain_maps=True")
        fingerprint = value_fingerprint(value)
        result = SearchResult([], 0, 0, 0)
        for map_state in self.maps + ([self._current] if self._current is not None else []):
            columns, true_matches, false_candidates, layers, positions = self.probe_map(
                map_state, value, fingerprint, collect_positions=True
            )
            result.positions.extend(positions)
            result.columns_read += columns
            result.false_candidates += false_candidates
            result.layers_consulted += layers
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def total_columns(self) -> int:
        return sum(self.columns_per_layer)

    def filter_bytes(self) -> int:
        """Bytes of the filter maps only (the index proper): every stored
        column takes `bytes_per_column` bytes. It excludes the raw event data
        (index entries), which a bloom-filter based client must keep too."""
        return self.total_columns() * self.p.bytes_per_column

    def header_bytes_per_block(self) -> int:
        """Bytes written in EVERY block header: one 32-byte Merkle root."""
        return HASH_SIZE


# ----------------------------------------------------------------------
# Size and proof formulas
# ----------------------------------------------------------------------
def tree_depth_per_row(p: IndexParameters) -> int:
    """Sibling hashes needed to prove one row of one map, from the root:
    1 (root split) + LOG2_EPOCH_HISTORY (select the epoch) + 1 (split
    GTI_FILTER_MAPS) + LOG2_MAPS_PER_EPOCH + LOG2_MAP_HEIGHT (select map and
    row inside the epoch). It depends only on the fixed parameters: the tree
    of the EIP has a bounded depth, independent of how much data it holds."""
    return 1 + p.log2_epoch_history + 1 + p.log2_maps_per_epoch + p.log2_map_height


def single_row_proof_bytes(p: IndexParameters, row_length: float) -> float:
    """Proof of the content of ONE row of ONE map: the hashes of the Merkle
    path plus the row data (`bytes_per_column` per stored column)."""
    return tree_depth_per_row(p) * HASH_SIZE + row_length * p.bytes_per_column


def full_history_proof_bytes(
    p: IndexParameters,
    num_maps: int,
    row_length: float,
    layers_consulted: float = 1.0,
    aggregated: bool = False,
) -> float:
    """Bytes to prove the answer of a search over the WHOLE history.

    naive: one independent proof per row consulted (`layers_consulted` rows
    per map on average).
    aggregated: ESTIMATE for a client that merges into one Merkle multiproof
    the rows with the same row index of the maps of an epoch (map_row_gti
    makes them adjacent in the tree): the path above the epoch is included
    once per epoch, plus about log2(MAPS_PER_EPOCH) extra hashes and the row
    data per map. It is an estimate based on the standard size of a Merkle
    multiproof over adjacent leaves, not a proof that was actually built.
    """
    if not aggregated:
        return num_maps * layers_consulted * single_row_proof_bytes(p, row_length)
    epochs = max(1, -(-num_maps // p.maps_per_epoch))
    shared = (tree_depth_per_row(p) - p.log2_maps_per_epoch) * HASH_SIZE
    extra_per_map = max(0, math.ceil(math.log2(p.maps_per_epoch))) * HASH_SIZE + row_length * p.bytes_per_column
    return layers_consulted * (epochs * shared + num_maps * extra_per_map)
