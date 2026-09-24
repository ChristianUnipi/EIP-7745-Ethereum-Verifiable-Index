"""
comparison.py
=============

Comparison of the EIP-7745 Log Index (eip7745_log_index.py) with the
logsBloom used today by Ethereum (bloom_filter_ethereum.py) on the real
"events" dataset (blocks 14,000,000-14,999,999, ~134.35 million unique event
occurrences per block, from https://zenodo.org/records/7957141).

File format ("events", checked byte by byte before writing the parser:
1,000,000 blocks, none truncated, reading ends exactly at end of file):
    blockId    uint32 big-endian (4 bytes)
    numEvents  uint32 big-endian (4 bytes)
    numEvents records of 52 bytes: contract address (20) + event signature (32)

Measurement protocol
--------------------
What is measured is the number of BYTES and the number of FALSE POSITIVES;
insertion time is deliberately not measured (it depends on the machine).

1. Query set. A sample of blocks is used to rank the values (addresses and
   topics) by frequency and to build four classes of queries:
     hot     the most frequent values (e.g. the `Transfer` signature),
     warm    values that recur a few dozen times,
     cold    values seen once in the sample (rare contracts),
     control random bytes that exist nowhere.
   Every query is then run against EVERY unit of the index, so the results
   are not an estimate from a small sample of blocks.
2. Log Index: the full dataset is streamed once per configuration. Every
   filter map is evaluated when it is completed (then discarded, so memory is
   bounded): for each query, the map is probed exactly as a client would
   (layer by layer, stopping at the first non-full row), and the confirmed
   matches and the *false candidates* (tag matched, entry is another value)
   are counted. Ground truth: the index counts the exact occurrences of the
   queried values in every map, and the result must match (recall check).
   When there are more maps than `--eval-maps` (e.g. 1M maps with
   VALUES_PER_MAP = 2^8) an equispaced subset of maps is evaluated.
3. Bloom filters: for every block (or every `--bloom-block-stride`-th block),
   each query is tested against the block's filter; the exact set of values
   of the block is the ground truth. Besides the real Ethereum filter
   (2048 bits, k=3) some generic filters with more bits are evaluated, in
   particular one with the same off-chain size per block as the Log Index.
4. Everything is normalised to the same quantity: expected false positives
   per query over the whole dataset (1M blocks), with a standard error
   computed from the variation across units (maps / blocks). The cost of a
   false positive is also reported in log entries that a client has to
   examine and discard: 1 entry per false candidate of the Log Index; all
   the events of the block per false positive block of the bloom filter.

Sizes count only the index (filter maps / bloom filters), not the raw log data
that every client has to keep in both cases. Proof sizes are analytic models
(see eip7745_log_index.py), not real Merkle multiproofs.

Usage
-----
    python3 comparison.py                        # full experiment
    python3 comparison.py --max-blocks 60000     # quick version
    python3 comparison.py --self-test            # synthetic sanity checks
"""

import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bloom_filter_ethereum import (
    ETHEREUM_BLOOM,
    BloomIndex,
    BloomSpec,
    bit_mask,
    theoretical_fp_probability,
    value_digest,
)
from eip7745_log_index import (
    HASH_SIZE,
    IndexParameters,
    LogIndex,
    full_history_proof_bytes,
    tree_depth_per_row,
    value_fingerprint,
)

FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVENTS_FILE = os.path.join(FOLDER, "events")
DEFAULT_RESULTS_DIR = os.path.join(FOLDER, "results")

BLOCK_HEADER = struct.Struct(">II")
EVENT_SIZE = 52  # 20-byte address + 32-byte topic

CLASSES = ("control", "cold", "warm", "hot")

# Crowdedness bins (events per block) for the bloom filter analysis.
DENSITY_EDGES = [50, 100, 200, 400]


def density_bin_of(n_events: int) -> int:
    for i, edge in enumerate(DENSITY_EDGES):
        if n_events < edge:
            return i
    return len(DENSITY_EDGES)


def density_bin_labels() -> List[str]:
    labels = [f"<{DENSITY_EDGES[0]}"]
    for a, b in zip(DENSITY_EDGES, DENSITY_EDGES[1:]):
        labels.append(f"{a}-{b - 1}")
    labels.append(f">={DENSITY_EDGES[-1]}")
    return labels

# The five VALUES_PER_MAP requested (2**16 is the real EIP value); all the
# other parameters stay at the values of the EIP. (log2_values_per_map, log2_map_width)
SWEEP_VALUES_PER_MAP = [(8, 24), (12, 24), (16, 24), (20, 24), (24, 24)]
# Same VALUES_PER_MAP, but LOG2_MAP_WIDTH grown with it so that the tag keeps
# 8 bits. With the default width, VALUES_PER_MAP = 2^24 leaves 0 tag bits.
SWEEP_CONSTANT_TAG_BITS = [(8, 16), (12, 20), (20, 28), (24, 32)]
# LOG2_MAP_WIDTH sweep at the original VALUES_PER_MAP = 2^16 (24 is the default).
SWEEP_MAP_WIDTH = [(16, 16), (16, 20), (16, 28)]

# Bloom filters compared. The Ethereum one is the real 2048-bit / k=3 filter;
# 6448 bits = 806 bytes is about the off-chain size per block of the Log Index
# with the default parameters (6 bytes per event x ~134 events per block).
BLOOM_SPECS = [
    ETHEREUM_BLOOM,
    BloomSpec("generic_4096_k3", 4096, 3),
    BloomSpec("generic_6448_k3", 6448, 3),
    BloomSpec("generic_6448_k8", 6448, 8),
    BloomSpec("generic_16384_k8", 16384, 8),
]


# ---------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------
def read_blocks(path: str, max_blocks: Optional[int] = None, stride: int = 1):
    """Generator of (block_id, [(address, topic), ...]).

    With stride > 1 only the blocks whose (0-based) reading index is a
    multiple of `stride` are returned; the others are skipped with a relative
    seek, without even reading their bytes."""
    with open(path, "rb") as f:
        read_index = 0
        while max_blocks is None or read_index < max_blocks:
            header = f.read(8)
            if len(header) < 8:
                break
            block_id, num_events = BLOCK_HEADER.unpack(header)
            payload_length = num_events * EVENT_SIZE
            if read_index % stride == 0:
                payload = f.read(payload_length)
                events = []
                for i in range(num_events):
                    offset = i * EVENT_SIZE
                    events.append((payload[offset : offset + 20], payload[offset + 20 : offset + 52]))
                yield block_id, events
            else:
                f.seek(payload_length, os.SEEK_CUR)
            read_index += 1


def scan_totals(path: str, max_blocks: Optional[int]) -> Tuple[int, int]:
    """(blocks, events) of the dataset (or of its first `max_blocks` blocks),
    reading only the block headers."""
    blocks = events = 0
    with open(path, "rb") as f:
        while max_blocks is None or blocks < max_blocks:
            header = f.read(8)
            if len(header) < 8:
                break
            _, num_events = BLOCK_HEADER.unpack(header)
            f.seek(num_events * EVENT_SIZE, os.SEEK_CUR)
            blocks += 1
            events += num_events
    return blocks, events


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------
@dataclass
class Query:
    value: bytes
    cls: str
    sample_frequency: int  # occurrences in the sample used to classify it


def build_queries(
    path: str,
    total_blocks: int,
    sample_blocks: int,
    n_hot: int,
    n_warm: int,
    n_cold: int,
    n_control: int,
    seed: int,
) -> List[Query]:
    """Rank the values of an equispaced sample of blocks by frequency and
    pick the queries of the four classes."""
    stride = max(1, total_blocks // sample_blocks)
    counter: Counter = Counter()
    for _, events in read_blocks(path, max_blocks=total_blocks, stride=stride):
        for address, topic in events:
            counter[address] += 1
            counter[topic] += 1

    rnd = random.Random(seed)
    ranked = counter.most_common()
    hot = ranked[:n_hot]
    rest = ranked[n_hot:]
    warm_pool = [item for item in rest if 10 <= item[1]]
    cold_pool = [item for item in rest if item[1] == 1]
    warm = rnd.sample(warm_pool, min(n_warm, len(warm_pool)))
    cold = rnd.sample(cold_pool, min(n_cold, len(cold_pool)))

    queries: List[Query] = []
    for cls, items in (("hot", hot), ("warm", warm), ("cold", cold)):
        for value, frequency in items:
            queries.append(Query(value, cls, frequency))
    while sum(1 for q in queries if q.cls == "control") < n_control:
        length = 20 if len(queries) % 2 == 0 else 32
        value = rnd.randbytes(length)
        if value not in counter:
            queries.append(Query(value, "control", 0))
    return queries


def save_queries(queries: List[Query], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"value": q.value.hex(), "class": q.cls, "sample_frequency": q.sample_frequency} for q in queries], f)


def load_queries(path: str) -> List[Query]:
    with open(path, "r", encoding="utf-8") as f:
        return [Query(bytes.fromhex(item["value"]), item["class"], item["sample_frequency"]) for item in json.load(f)]


# ---------------------------------------------------------------------------
# Log Index: evaluation of every map
# ---------------------------------------------------------------------------
def _new_accumulators(queries: List[Query]) -> Dict[str, dict]:
    accumulators = {}
    for cls in CLASSES:
        accumulators[cls] = {
            "n_queries": sum(1 for q in queries if q.cls == cls),
            "units": 0,  # maps evaluated
            "sum_fp": 0,  # sum over maps of the false candidates of the class
            "sum_fp2": 0,  # sum of squares (for the standard error)
            "sum_tp": 0,  # confirmed matches
            "sum_columns": 0,  # columns read
            "sum_layers": 0,  # rows consulted (one per layer, per map, per query)
        }
    return accumulators


class MapEvaluator:
    """Callback for `LogIndex.on_map_complete`: probes every query in the
    completed map and accumulates the counts per class."""

    def __init__(self, queries: List[Query], map_stride: int) -> None:
        self.map_stride = map_stride
        self.values = [q.value for q in queries]
        self.fingerprints = [value_fingerprint(q.value) for q in queries]
        self.class_of = [CLASSES.index(q.cls) for q in queries]
        self.tracked = {q.value: i for i, q in enumerate(queries)}
        self.accumulators = _new_accumulators(queries)
        self.recall_checks = 0
        self.recall_mismatches = 0
        # rows of the queries for the current masked-map group, per layer
        self._row_cache: List[Dict[int, int]] = []
        self._row_group: List[int] = []

    def on_map_complete(self, index: LogIndex, map_state) -> None:
        if map_state.index % self.map_stride:
            return
        frequencies = index.p.mapping_frequency_by_layer
        if not self._row_cache:
            self._row_cache = [{} for _ in frequencies]
            self._row_group = [-1] * len(frequencies)
        masked = [map_state.index - map_state.index % f for f in frequencies]
        for layer, masked_map in enumerate(masked):
            if masked_map != self._row_group[layer]:
                self._row_cache[layer].clear()
                self._row_group[layer] = masked_map

        current = [0]

        def row_provider(layer: int) -> int:
            i = current[0]
            cache = self._row_cache[layer]
            row = cache.get(i)
            if row is None:
                row = index.row_index_for_masked_map(self.values[i], masked[layer], layer)
                cache[i] = row
            return row

        n_classes = len(CLASSES)
        fp = [0] * n_classes
        tp = [0] * n_classes
        columns = [0] * n_classes
        layers = [0] * n_classes
        truth = map_state.tracked_counts
        for i in range(len(self.values)):
            current[0] = i
            columns_read, true_matches, false_candidates, layers_consulted, _ = index.probe_map(
                map_state, self.values[i], self.fingerprints[i], row_provider
            )
            c = self.class_of[i]
            fp[c] += false_candidates
            tp[c] += true_matches
            columns[c] += columns_read
            layers[c] += layers_consulted
            self.recall_checks += 1
            if true_matches != truth.get(i, 0):
                self.recall_mismatches += 1

        for c, cls in enumerate(CLASSES):
            acc = self.accumulators[cls]
            acc["units"] += 1
            acc["sum_fp"] += fp[c]
            acc["sum_fp2"] += fp[c] * fp[c]
            acc["sum_tp"] += tp[c]
            acc["sum_columns"] += columns[c]
            acc["sum_layers"] += layers[c]


def run_log_index_worker(
    events_file: str,
    parameters: IndexParameters,
    max_blocks: Optional[int],
    total_insertions: int,
    queries: List[Query],
    eval_maps: int,
    output_path: str,
) -> None:
    expected_maps = max(1, -(-total_insertions // parameters.values_per_map))
    map_stride = max(1, -(-expected_maps // eval_maps))
    evaluator = MapEvaluator(queries, map_stride)
    index = LogIndex(
        parameters,
        retain_maps=False,
        on_map_complete=evaluator.on_map_complete,
        tracked_values=evaluator.tracked,
    )
    blocks = events = 0
    t0 = time.perf_counter()
    for _, block_events in read_blocks(events_file, max_blocks):
        for address, topic in block_events:
            index.insert_event(address, topic)
        blocks += 1
        events += len(block_events)
    index.finish()
    duration = time.perf_counter() - t0

    evaluated_maps = evaluator.accumulators["control"]["units"]
    result = {
        "kind": "log_index",
        "log2_values_per_map": parameters.log2_values_per_map,
        "log2_map_width": parameters.log2_map_width,
        "log2_map_height": parameters.log2_map_height,
        "tag_bits": parameters.tag_bits,
        "bytes_per_column": parameters.bytes_per_column,
        "blocks": blocks,
        "events": events,
        "insertions": index.next_entry,
        "num_maps": index.num_maps,
        "map_stride": map_stride,
        "evaluated_maps": evaluated_maps,
        "columns_per_layer": index.columns_per_layer,
        "entries_beyond_last_layer": index.entries_beyond_last_layer,
        "filter_bytes": index.filter_bytes(),
        "header_bytes_per_block": index.header_bytes_per_block(),
        "classes": evaluator.accumulators,
        "recall_checks": evaluator.recall_checks,
        "recall_mismatches": evaluator.recall_mismatches,
        "duration_seconds": duration,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


# ---------------------------------------------------------------------------
# Bloom filters: evaluation of every (sampled) block
# ---------------------------------------------------------------------------
def run_bloom_worker(
    events_file: str,
    max_blocks: Optional[int],
    stride: int,
    total_blocks: int,
    queries: List[Query],
    specs: List[BloomSpec],
    output_path: str,
) -> None:
    query_values = [q.value for q in queries]
    class_of = [CLASSES.index(q.cls) for q in queries]
    query_index = {value: i for i, value in enumerate(query_values)}
    query_digests = [value_digest(v) for v in query_values]
    query_masks = [[bit_mask(spec, d) for d in query_digests] for spec in specs]

    n_classes = len(CLASSES)
    per_spec = []
    for spec in specs:
        accs = {}
        for cls in CLASSES:
            accs[cls] = {
                "n_queries": sum(1 for q in queries if q.cls == cls),
                "units": 0,  # blocks evaluated
                "sum_fp": 0,
                "sum_fp2": 0,
                "sum_tp": 0,
                "sum_wasted": 0,  # false positives x events of the block (entries to examine)
                "sum_wasted2": 0,
            }
        per_spec.append({
            "spec": spec,
            "accs": accs,
            "false_negatives": 0,
            # false positives of the control class by crowdedness of the block
            "density": [{"blocks": 0, "fp_control": 0} for _ in range(len(DENSITY_EDGES) + 1)],
        })
    control_class = CLASSES.index("control")

    blocks_evaluated = 0
    events_evaluated = 0
    t0 = time.perf_counter()
    for _, events in read_blocks(events_file, max_blocks, stride):
        values = set()
        for address, topic in events:
            values.add(address)
            values.add(topic)
        digests = [(v, value_digest(v)) for v in values]
        n_events = len(events)
        present_queries = [query_index[v] for v in values if v in query_index]

        for s, entry in enumerate(per_spec):
            spec = entry["spec"]
            bits = 0
            for _, digest in digests:
                bits |= bit_mask(spec, digest)
            masks = query_masks[s]

            fp = [0] * n_classes
            tp = [0] * n_classes
            for i in range(len(masks)):
                mask = masks[i]
                if bits & mask == mask:
                    if query_values[i] in values:
                        tp[class_of[i]] += 1
                    else:
                        fp[class_of[i]] += 1
            for i in present_queries:
                if bits & masks[i] != masks[i]:
                    entry["false_negatives"] += 1
            density_bin = density_bin_of(n_events)
            entry["density"][density_bin]["blocks"] += 1
            entry["density"][density_bin]["fp_control"] += fp[control_class]
            for c, cls in enumerate(CLASSES):
                acc = entry["accs"][cls]
                acc["units"] += 1
                acc["sum_fp"] += fp[c]
                acc["sum_fp2"] += fp[c] * fp[c]
                acc["sum_tp"] += tp[c]
                wasted = fp[c] * n_events
                acc["sum_wasted"] += wasted
                acc["sum_wasted2"] += wasted * wasted
        blocks_evaluated += 1
        events_evaluated += n_events
    duration = time.perf_counter() - t0

    result = {
        "kind": "bloom",
        "stride": stride,
        "total_blocks": total_blocks,
        "blocks_evaluated": blocks_evaluated,
        "avg_events_per_block": events_evaluated / max(1, blocks_evaluated),
        "specs": {
            entry["spec"].name: {
                "m_bits": entry["spec"].m_bits,
                "k": entry["spec"].k,
                "size_bytes": entry["spec"].size_bytes,
                "classes": entry["accs"],
                "false_negatives": entry["false_negatives"],
                "density": entry["density"],
            }
            for entry in per_spec
        },
        "duration_seconds": duration,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


# ---------------------------------------------------------------------------
# Orchestration: parallel workers
# ---------------------------------------------------------------------------
def config_tag(log2_values_per_map: int, log2_map_width: int) -> str:
    return f"v{log2_values_per_map}_w{log2_map_width}"


def all_log_index_configs(skip_extra: bool) -> List[Tuple[int, int]]:
    configs = list(SWEEP_VALUES_PER_MAP)
    if not skip_extra:
        configs += SWEEP_CONSTANT_TAG_BITS + SWEEP_MAP_WIDTH
    return configs


def run_workers(tasks: List[Tuple[str, str, List[str]]], max_processes: int) -> None:
    """Run the worker commands in parallel processes (at most `max_processes`
    at a time). Each worker writes its own JSON file. Tasks whose output file
    already exists are skipped, so an interrupted run can be resumed."""
    pending = []
    for tag, output_path, command in tasks:
        if os.path.exists(output_path):
            print(f"  [skip]  {tag} (result already present)")
        else:
            pending.append((tag, output_path, command))

    running: List[Tuple[str, str, subprocess.Popen]] = []
    failures = []
    t0 = time.perf_counter()
    while pending or running:
        while pending and len(running) < max_processes:
            tag, output_path, command = pending.pop(0)
            print(f"  [start] {tag}", flush=True)
            running.append((tag, output_path, subprocess.Popen(command)))
        still_running = []
        for tag, output_path, process in running:
            code = process.poll()
            if code is None:
                still_running.append((tag, output_path, process))
            elif code != 0:
                failures.append(tag)
                print(f"  [FAIL]  {tag} (exit code {code})", flush=True)
            else:
                print(f"  [done]  {tag}  ({time.perf_counter() - t0:.0f}s since start)", flush=True)
        running = still_running
        if running:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"workers failed: {failures}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def format_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:,.2f} PiB"


def fmt(value: float, se: Optional[float] = None) -> str:
    def one(x: float) -> str:
        if x == 0:
            return "0"
        if abs(x) >= 100:
            return f"{x:,.0f}"
        if abs(x) >= 1:
            return f"{x:,.2f}"
        return f"{x:.3g}"

    if se is None or se == 0:
        return one(value)
    return f"{one(value)} ± {one(se)}"


def per_query_per_unit(acc: dict, key: str = "sum_fp", key2: str = "sum_fp2") -> Tuple[float, float]:
    """Mean and standard error of `key` per query per unit (map or block)."""
    n = acc["units"]
    n_queries = acc["n_queries"]
    if n == 0 or n_queries == 0:
        return 0.0, 0.0
    mean = acc[key] / n
    variance = max(0.0, (acc[key2] - n * mean * mean) / (n - 1)) if n > 1 else 0.0
    return mean / n_queries, math.sqrt(variance / n) / n_queries


def fp_full_history(acc: dict, total_units: int, key: str = "sum_fp", key2: str = "sum_fp2") -> Tuple[float, float]:
    """Expected false positives (or wasted entries) per query over the whole dataset."""
    mean, se = per_query_per_unit(acc, key, key2)
    return mean * total_units, se * total_units


def precision(acc: dict) -> Optional[float]:
    total = acc["sum_tp"] + acc["sum_fp"]
    return acc["sum_tp"] / total if total else None


def ratio_text(baseline: float, value: float) -> str:
    if value == 0:
        return "no false positive observed"
    return f"{baseline / value:,.1f}x fewer"


def build_report(results_dir: str, scan: dict, queries: List[Query], settings: dict) -> str:
    total_blocks = scan["blocks"]
    total_events = scan["events"]

    li: Dict[str, dict] = {}
    for name in os.listdir(results_dir):
        if name.startswith("li_") and name.endswith(".json"):
            with open(os.path.join(results_dir, name), "r", encoding="utf-8") as f:
                data = json.load(f)
            li[config_tag(data["log2_values_per_map"], data["log2_map_width"])] = data
    with open(os.path.join(results_dir, "bloom.json"), "r", encoding="utf-8") as f:
        bloom = json.load(f)

    default_tag = config_tag(16, 24)
    li_default = li[default_tag]
    eth = bloom["specs"][ETHEREUM_BLOOM.name]
    params_default = IndexParameters()

    out: List[str] = []
    w = out.append
    w("# EIP-7745 Log Index vs Ethereum logsBloom\n")
    w(f"Dataset: {total_blocks:,} blocks, {total_events:,} unique event occurrences "
      f"({2 * total_events:,} index insertions: one for the address and one for the topic).\n")
    counts = {cls: sum(1 for q in queries if q.cls == cls) for cls in CLASSES}
    w("Queries: " + ", ".join(f"{counts[c]} {c}" for c in CLASSES) +
      f". Log Index: every map evaluated (or an equispaced subset of at least {settings['eval_maps']} maps "
      f"when there are more); bloom filters: 1 block out of {bloom['stride']} "
      f"({bloom['blocks_evaluated']:,} blocks).\n")
    w("Definitions. *False positive* (Log Index): a stored column whose tag matches but whose entry is a "
      "different value (a *false candidate*: the client would fetch the entry and discard it). *False positive* "
      "(bloom): a block whose filter says \"maybe\" but that does not contain the value. Both are reported as the "
      "expected number per query over the whole dataset, with the standard error across maps / blocks. "
      "Recall is 100% for both by construction and is verified (last table).\n")

    # ---------------- sizes
    w("## 1. Size\n")
    w("| | Ethereum logsBloom | Log Index (EIP defaults) |")
    w("|---|---|---|")
    w(f"| Bytes in every block header | {eth['size_bytes']} (the filter) | {li_default['header_bytes_per_block']} (one Merkle root) |")
    w(f"| Index size (whole dataset) | {format_bytes(eth['size_bytes'] * total_blocks)} | {format_bytes(li_default['filter_bytes'])} |")
    w(f"| Per block | {eth['size_bytes']} B | {li_default['filter_bytes'] / total_blocks:,.1f} B |")
    w(f"| Per event | {eth['size_bytes'] * total_blocks / total_events:.2f} B | {li_default['filter_bytes'] / total_events:.2f} B |")
    w("\nThe Log Index stores one column of `LOG2_MAP_WIDTH/8` bytes per insertion, so its size depends on the "
      "column width and on the number of events, never on VALUES_PER_MAP. The Ethereum filter is fixed at 256 B "
      "per block, whatever the block contains. The Log Index index is therefore larger off-chain, but only 32 B per "
      "block have to be in the header (and in consensus).\n")

    # ---------------- headline FP
    w("## 2. False positives per query over the whole dataset (headline)\n")
    w("Ethereum logsBloom (real 2048-bit filter) vs Log Index with the parameters of the EIP.\n")
    w("| Query class | bloom: false blocks | Log Index: false candidates | Reduction | bloom precision | Log Index precision |")
    w("|---|---|---|---|---|---|")
    for cls in CLASSES:
        b, b_se = fp_full_history(eth["classes"][cls], total_blocks)
        l, l_se = fp_full_history(li_default["classes"][cls], li_default["num_maps"])
        pb = precision(eth["classes"][cls]) if cls != "control" else None
        pl = precision(li_default["classes"][cls]) if cls != "control" else None
        w(f"| {cls} | {fmt(b, b_se)} | {fmt(l, l_se)} | {ratio_text(b, l)} | "
          f"{'-' if pb is None else f'{pb:.4f}'} | {'-' if pl is None else f'{pl:.4f}'} |")
    w("\nprecision = confirmed / (confirmed + false), over every unit evaluated (blocks for the bloom filter, "
      "maps for the Log Index); it is not defined (-) for the control class, which has no true matches.\n")

    w("### Work wasted on false positives (log entries examined and discarded per query)\n")
    w("A false block of the bloom filter forces the client to re-read ALL the events of the block; a false candidate "
      "of the Log Index costs one entry. False blocks are biased towards crowded blocks, so the average is weighted by "
      "the real number of events.\n")
    w("| Query class | bloom: entries examined | Log Index: entries examined | Reduction |")
    w("|---|---|---|---|")
    for cls in CLASSES:
        wb, wb_se = fp_full_history(eth["classes"][cls], total_blocks, "sum_wasted", "sum_wasted2")
        l, l_se = fp_full_history(li_default["classes"][cls], li_default["num_maps"])
        w(f"| {cls} | {fmt(wb, wb_se)} | {fmt(l, l_se)} | {ratio_text(wb, l)} |")

    # ---------------- density
    w("\n### False positives of the bloom filter depend on how crowded the block is\n")
    w("Probability that a random absent value (control class) is flagged, per block and per query, by the number "
      "of events in the block. A Log Index map always holds exactly VALUES_PER_MAP entries, so its rate does not "
      "depend on the traffic of any block.\n")
    labels = density_bin_labels()
    w("| Structure | " + " | ".join(f"{label} events" for label in labels) + " | all blocks |")
    w("|---|" + "---|" * (len(labels) + 1))
    for name, spec in bloom["specs"].items():
        n_control = spec["classes"]["control"]["n_queries"]
        cells = []
        for b in spec["density"]:
            cells.append(fmt(b["fp_control"] / (b["blocks"] * n_control)) if b["blocks"] else "-")
        overall = spec["classes"]["control"]["sum_fp"] / max(1, spec["classes"]["control"]["units"] * n_control)
        w(f"| bloom {name} | " + " | ".join(cells) + f" | {fmt(overall)} |")
    eth_blocks = [b["blocks"] for b in eth["density"]]
    total_eval = max(1, sum(eth_blocks))
    w("| (share of the blocks evaluated) | " + " | ".join(f"{100 * n / total_eval:.1f}%" for n in eth_blocks) + " | 100% |")
    li_control, _ = fp_full_history(li_default["classes"]["control"], li_default["num_maps"])
    w(f"| **Log Index (EIP defaults)** | " + " | ".join([fmt(li_control / total_blocks)] * len(labels)) +
      f" | {fmt(li_control / total_blocks)} |")

    # ---------------- parity
    w("\n## 3. Same size: bigger bloom filters vs the Log Index\n")
    w("What if the bloom filter had as many bytes as the Log Index? Expected false blocks per query over the "
      "whole dataset. Only the 256-byte filter fits the block header today.\n")
    w("| Structure | Bytes per block | control | cold | warm | hot |")
    w("|---|---|---|---|---|---|")
    for name, spec in bloom["specs"].items():
        cells = []
        for cls in ("control", "cold", "warm", "hot"):
            v, se = fp_full_history(spec["classes"][cls], total_blocks)
            cells.append(fmt(v, se))
        w(f"| bloom {name} | {spec['size_bytes']} | " + " | ".join(cells) + " |")
    cells = []
    for cls in ("control", "cold", "warm", "hot"):
        v, se = fp_full_history(li_default["classes"][cls], li_default["num_maps"])
        cells.append(fmt(v, se))
    w(f"| **Log Index (EIP defaults)** | {li_default['filter_bytes'] / total_blocks:,.0f} | " + " | ".join(cells) + " |")
    w("\nThis is the fair test of the design, and its outcome is whatever the numbers above say: a bloom filter "
      "with as many bytes and enough hash functions can match the Log Index on false positives. What only the "
      "Log Index offers is the 32-byte header (a 806-byte bloom filter cannot be put in every header), the "
      "compact proofs (section 4) and a false positive rate that does not depend on how crowded a block is "
      "(previous table).\n")

    # ---------------- proofs
    w("## 4. Bytes to search the whole history (modelled)\n")
    w("Log Index: proof of the rows consulted in every map (analytic model, see eip7745_log_index.py) "
      "using the average number of layers and columns actually observed for a rare value (control class). "
      "Bloom: the filter of every block must be read.\n")
    w("| Structure | Bytes for a full-history search |")
    w("|---|---|")
    w(f"| bloom {ETHEREUM_BLOOM.name} | {format_bytes(eth['size_bytes'] * total_blocks)} |")
    ctrl = li_default["classes"]["control"]
    lookups = max(1, ctrl["units"] * ctrl["n_queries"])
    layers = ctrl["sum_layers"] / lookups
    row_length = ctrl["sum_columns"] / max(1, ctrl["sum_layers"])
    naive = full_history_proof_bytes(params_default, li_default["num_maps"], row_length, layers, aggregated=False)
    aggregated = full_history_proof_bytes(params_default, li_default["num_maps"], row_length, layers, aggregated=True)
    w(f"| Log Index, one proof per row (no aggregation) | {format_bytes(naive)} |")
    w(f"| Log Index, aggregated multiproof (estimate) | {format_bytes(aggregated)} |")
    w(f"\nAverage rows consulted per map: {layers:.3f}; average columns per row: {row_length:.3f}; "
      f"maps: {li_default['num_maps']:,}; proof depth per row: {tree_depth_per_row(params_default)} hashes of {HASH_SIZE} B.\n")

    # ---------------- sweeps
    def sweep_table(title: str, description: str, configs: List[Tuple[int, int]]) -> None:
        w(f"## {title}\n")
        w(description + "\n")
        w("| VALUES_PER_MAP | MAP_WIDTH | tag bits | maps | filter B/block | FP control | FP cold | FP warm | FP hot | "
          "beyond last layer | rows/map (control) | full-history proof, aggregated |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for l2v, l2w in configs:
            tag = config_tag(l2v, l2w)
            if tag not in li:
                continue
            d = li[tag]
            cells = []
            for cls in ("control", "cold", "warm", "hot"):
                v, se = fp_full_history(d["classes"][cls], d["num_maps"])
                cells.append(fmt(v, se))
            c = d["classes"]["control"]
            lk = max(1, c["units"] * c["n_queries"])
            lay = c["sum_layers"] / lk
            rl = c["sum_columns"] / max(1, c["sum_layers"])
            p = IndexParameters(log2_values_per_map=l2v, log2_map_width=l2w)
            agg = full_history_proof_bytes(p, d["num_maps"], rl, lay, aggregated=True)
            beyond = d["entries_beyond_last_layer"] / d["insertions"]
            w(f"| 2^{l2v} | {l2w} | {d['tag_bits']} | {d['num_maps']:,} | {d['filter_bytes'] / total_blocks:,.0f} | "
              + " | ".join(cells) + f" | {beyond:.2%} | {lay:.2f} | {format_bytes(agg)} |")
        w("")

    sweep_table(
        "5. Sweep of VALUES_PER_MAP (all other parameters at the EIP values)",
        "False positives (FP) are expected false candidates per query over the whole dataset. With the default "
        "LOG2_MAP_WIDTH = 24 the tag has 24 - log2(VALUES_PER_MAP) bits, so it disappears at 2^24.",
        SWEEP_VALUES_PER_MAP,
    )
    sweep_table(
        "6. Sweep of VALUES_PER_MAP keeping 8 tag bits",
        "LOG2_MAP_WIDTH grows with VALUES_PER_MAP so that every column keeps an 8-bit tag; columns become wider "
        "(bigger index) but the false positive rate per column read stays constant.",
        [(8, 16), (12, 20), (16, 24), (20, 28), (24, 32)],
    )
    sweep_table(
        "7. Sweep of LOG2_MAP_WIDTH at VALUES_PER_MAP = 2^16",
        "More bits per column mean more tag bits: fewer false positives, larger index.",
        [(16, 16), (16, 20), (16, 24), (16, 28)],
    )

    # ---------------- sanity
    w("## 8. Sanity checks\n")
    w("| Configuration | maps evaluated / total | recall checks | recall mismatches |")
    w("|---|---|---|---|")
    for tag in sorted(li):
        d = li[tag]
        w(f"| {tag} | {d['evaluated_maps']:,} / {d['num_maps']:,} | {d['recall_checks']:,} | {d['recall_mismatches']} |")
    fn = {name: s["false_negatives"] for name, s in bloom["specs"].items()}
    w(f"\nBloom false negatives (must be 0): {fn}\n")
    w("Theoretical false positive probability of the Ethereum filter for a block with n distinct values "
      f"(n=100: {theoretical_fp_probability(ETHEREUM_BLOOM, 100):.2e}, n=250: {theoretical_fp_probability(ETHEREUM_BLOOM, 250):.2e}, "
      f"n=500: {theoretical_fp_probability(ETHEREUM_BLOOM, 500):.2e}, n=1000: {theoretical_fp_probability(ETHEREUM_BLOOM, 1000):.2e}); "
      f"average events per block in the dataset: {bloom['avg_events_per_block']:.1f}.\n")
    w("## Caveats\n")
    w("* Sizes count only the index, not the raw log data; the real EIP also stores the index entries in its tree.\n"
      "* Proof sizes are analytic models, not real Merkle multiproofs.\n"
      "* The hash functions (BLAKE2s for rows/tags, SHA3-256 for bloom filters) are not the ones of Ethereum; "
      "the false positive rates depend on the distribution of the hashes, not on which hash is used.\n"
      "* Confirmation of a candidate uses a 64-bit fingerprint of the value instead of the value itself; the recall "
      "check against exact ground truth would reveal any collision.\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def self_test() -> None:
    """Synthetic checks: recall of the Log Index and of the bloom filters, and
    consistency of the evaluator, on a small dataset with heavy overflow."""
    rnd = random.Random(7)
    addresses = [rnd.randbytes(20) for _ in range(40)]
    topics = [rnd.randbytes(32) for _ in range(6)]
    parameters = IndexParameters(
        log2_values_per_map=6,
        log2_map_height=3,
        log2_map_width=10,
        mapping_frequency_by_layer=(4, 2, 1, 1),
        max_row_length_by_layer=(2, 4, 8, 16),
    )

    queries = [Query(v, "hot", 0) for v in addresses[:5] + topics[:2]]
    queries += [Query(v, "cold", 0) for v in addresses[5:12]]
    queries += [Query(rnd.randbytes(20), "control", 0) for _ in range(30)]
    evaluator = MapEvaluator(queries, map_stride=1)
    index = LogIndex(parameters, retain_maps=True, on_map_complete=evaluator.on_map_complete,
                     tracked_values=evaluator.tracked)
    bloom_index = BloomIndex()
    positions: Dict[bytes, List[int]] = {}
    for block_id in range(300):
        for _ in range(rnd.randint(1, 8)):
            address = rnd.choice(addresses) if rnd.random() < 0.9 else rnd.randbytes(20)
            topic = rnd.choice(topics)
            a, t = index.insert_event(address, topic)
            positions.setdefault(address, []).append(a)
            positions.setdefault(topic, []).append(t)
            bloom_index.add_event(block_id, address, topic)
    index.finish()

    assert index.entries_beyond_last_layer >= 0
    assert sum(index.columns_per_layer) == index.next_entry
    assert index.columns_per_layer[1] > 0, "the test must exercise the overflow to layer 1"

    for value in list(positions)[:60]:
        result = index.search(value)
        assert sorted(result.positions) == sorted(positions[value]), "missed or extra positions"
    for _ in range(100):
        assert index.search(rnd.randbytes(20)).positions == []
    assert evaluator.recall_mismatches == 0, evaluator.recall_mismatches
    assert evaluator.recall_checks > 0

    for value in list(positions)[:60]:
        assert bloom_index.search(value).confirmed_blocks, "bloom false negative"
    print("self-test OK "
          f"({index.next_entry} entries, {index.num_maps} maps, columns per layer {index.columns_per_layer}, "
          f"{evaluator.recall_checks} recall checks)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-file", default=DEFAULT_EVENTS_FILE)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--max-blocks", type=int, default=None, help="use only the first N blocks (default: all)")
    ap.add_argument("--eval-maps", type=int, default=3000, help="maps evaluated per configuration (at most)")
    ap.add_argument("--bloom-block-stride", type=int, default=10, help="evaluate one block out of N for the bloom filters")
    ap.add_argument("--query-sample-blocks", type=int, default=2000)
    ap.add_argument("--hot", type=int, default=100)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--cold", type=int, default=600)
    ap.add_argument("--control", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-processes", type=int, default=None)
    ap.add_argument("--skip-extra-sweeps", action="store_true")
    ap.add_argument("--self-test", action="store_true")

    # internal: used by the child processes
    ap.add_argument("--worker", choices=["log_index", "bloom"], help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", help=argparse.SUPPRESS)
    ap.add_argument("--queries-file", help=argparse.SUPPRESS)
    ap.add_argument("--total-insertions", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--total-blocks", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--log2-values-per-map", type=int, default=16, help=argparse.SUPPRESS)
    ap.add_argument("--log2-map-width", type=int, default=24, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.worker == "log_index":
        parameters = IndexParameters(log2_values_per_map=args.log2_values_per_map, log2_map_width=args.log2_map_width)
        run_log_index_worker(args.events_file, parameters, args.max_blocks, args.total_insertions,
                             load_queries(args.queries_file), args.eval_maps, args.worker_output)
        return
    if args.worker == "bloom":
        run_bloom_worker(args.events_file, args.max_blocks, args.bloom_block_stride, args.total_blocks,
                         load_queries(args.queries_file), BLOOM_SPECS, args.worker_output)
        return

    os.makedirs(args.results_dir, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 78)
    print("STEP 1: dataset scan and query set")
    print("=" * 78)
    scan_path = os.path.join(args.results_dir, "scan.json")
    if os.path.exists(scan_path):
        with open(scan_path, "r", encoding="utf-8") as f:
            scan = json.load(f)
    else:
        blocks, events = scan_totals(args.events_file, args.max_blocks)
        scan = {"blocks": blocks, "events": events, "insertions": 2 * events}
        with open(scan_path, "w", encoding="utf-8") as f:
            json.dump(scan, f)
    print(f"  {scan['blocks']:,} blocks, {scan['events']:,} events, {scan['insertions']:,} insertions")

    queries_path = os.path.join(args.results_dir, "queries.json")
    if os.path.exists(queries_path):
        queries = load_queries(queries_path)
    else:
        queries = build_queries(args.events_file, scan["blocks"], args.query_sample_blocks,
                                args.hot, args.warm, args.cold, args.control, args.seed)
        save_queries(queries, queries_path)
    print("  queries: " + ", ".join(f"{sum(1 for q in queries if q.cls == c)} {c}" for c in CLASSES))

    print("\n" + "=" * 78)
    print("STEP 2: bloom filters and Log Index configurations (parallel workers)")
    print("=" * 78)
    common = ["--events-file", args.events_file, "--queries-file", queries_path,
              "--eval-maps", str(args.eval_maps)]
    if args.max_blocks is not None:
        common += ["--max-blocks", str(args.max_blocks)]
    script = os.path.abspath(__file__)

    tasks = []
    bloom_output = os.path.join(args.results_dir, "bloom.json")
    tasks.append(("bloom", bloom_output,
                  [sys.executable, script, "--worker", "bloom", "--worker-output", bloom_output,
                   "--total-blocks", str(scan["blocks"]), "--bloom-block-stride", str(args.bloom_block_stride)] + common))
    for l2v, l2w in all_log_index_configs(args.skip_extra_sweeps):
        output = os.path.join(args.results_dir, f"li_{config_tag(l2v, l2w)}.json")
        tasks.append((f"log_index {config_tag(l2v, l2w)}", output,
                      [sys.executable, script, "--worker", "log_index", "--worker-output", output,
                       "--total-insertions", str(scan["insertions"]),
                       "--log2-values-per-map", str(l2v), "--log2-map-width", str(l2w)] + common))
    run_workers(tasks, args.max_processes or max(1, os.cpu_count() or 1))

    print("\n" + "=" * 78)
    print("STEP 3: report")
    print("=" * 78)
    report = build_report(args.results_dir, scan, queries, {"eval_maps": args.eval_maps})
    report_path = os.path.join(args.results_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nReport saved to: {report_path}")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
