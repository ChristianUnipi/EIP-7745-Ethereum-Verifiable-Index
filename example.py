import hashlib
from typing import List, Optional


def _hash(data: bytes) -> bytes: return hashlib.sha256(data).digest()


def merkle_root(leaves: List[bytes]) -> bytes:
    if not leaves: return _hash(b"")

    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # duplicate the last odd node

        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(_hash(level[i] + level[i + 1]))
        level = next_level

    return level[0]


class LogIndex:
    """

    Components:
      - self.entries: the actual sequential log (equivalent to the EIP's
        "index entries"): every inserted string is appended at the end and
        its position (its index in the array) becomes its permanent
        identifier.
      - self.entries_per_epoch: how many consecutive entries share the same
        filter map (equivalent to VALUES_PER_MAP in the real EIP). In the
        real thing it is a large number (tens of thousands); here it is
        small to make the switch from one epoch to the next visible.
      - self.mapping_frequency_by_layer: a grouping frequency of the epochs
        for each layer (equivalent to 2 ** LOG2_MAPPING_FREQUENCY in the
        real EIP, where it is [1024, 64, 4, 1]). The higher the frequency
        of a layer, the more consecutive epochs share the same row at that
        layer.
      - self.max_row_length_by_layer: how many columns a row of each layer
        can hold at most before the excess entries have to move to the next
        layer (equivalent to MAX_ROW_LENGTH in the real EIP, where it is
        [8, 168, 2728, 10920]).
      - self.maps: one filter map per epoch (equivalent to the EIP's
        "filter maps"). self.maps[epoch][row] is the list of COLUMNS (plain
        integers, no position) recorded in that row of that epoch: the
        structure never stores explicitly "this column belongs to this
        position", exactly like the real EIP. It is up to the search to
        reconstruct it by trying the candidate positions (see search()).

    The values used here for mapping_frequency_by_layer and
    max_row_length_by_layer are much smaller than those of the EIP, so that
    they stay visible with only a few dozen example entries: the formula and
    the mechanism (several layers, each with its own grouping frequency and
    its own capacity, with a move to the next layer when a row is full) are
    however identical.
    """

    def __init__(
        self,
        entries_per_epoch: int = 5,
        number_of_rows: int = 4,
        column_bits: int = 6,
        mapping_frequency_by_layer: Optional[List[int]] = None,
        max_row_length_by_layer: Optional[List[int]] = None,
    ):
        self.entries: List[str] = []
        self.entries_per_epoch = entries_per_epoch
        self.number_of_rows = number_of_rows
        self.column_bits = column_bits

        if mapping_frequency_by_layer is None:
            mapping_frequency_by_layer = [4, 2, 1, 1]
        if max_row_length_by_layer is None:
            max_row_length_by_layer = [2, 4, 8, 16]
        self.mapping_frequency_by_layer = mapping_frequency_by_layer
        self.max_row_length_by_layer = max_row_length_by_layer

        # One map per epoch; every map is a list of 'number_of_rows'
        # rows; every row is a list of columns (integers).
        self.maps: List[List[List[int]]] = []
        self._merkle_leaves: List[bytes] = []
        # For illustration only (not part of the mechanism): keeps track of
        # the layer in which every inserted entry actually ended up, useful
        # to show in the demo when a move to another layer happens because
        # of a full row.
        self._layer_of_each_entry: List[int] = []

    def _row(self, value: str, epoch: int, layer_index: int) -> int:

        mapping_frequency = self.mapping_frequency_by_layer[layer_index]
        masked_epoch = epoch - (epoch % mapping_frequency)
        data = (
            value.encode("utf-8")
            + masked_epoch.to_bytes(4, byteorder="big")
            + layer_index.to_bytes(4, byteorder="big")
        )
        integer_hash = int.from_bytes(_hash(data), byteorder="big")
        return integer_hash % self.number_of_rows

    def _column(self, value: str, position: int) -> int:

        data = value.encode("utf-8") + position.to_bytes(4, byteorder="big")
        integer_hash = int.from_bytes(_hash(data), byteorder="big")
        return integer_hash % (1 << self.column_bits)

    def insert(self, value: str) -> int:
        position = len(self.entries)
        self.entries.append(value)

        epoch = position // self.entries_per_epoch
        if epoch == len(self.maps):
            # First entry of a new epoch: a new filter map is opened,
            # made of 'number_of_rows' initially empty rows.
            new_map = []
            for _ in range(self.number_of_rows):
                new_map.append([])
            self.maps.append(new_map)

        column = self._column(value, position)

        layer_index = 0
        while True:
            row = self._row(value, epoch, layer_index)
            row_columns = self.maps[epoch][row]
            last_available_layer = len(self.max_row_length_by_layer) - 1
            max_capacity = self.max_row_length_by_layer[
                min(layer_index, last_available_layer)
            ]
            if len(row_columns) < max_capacity:
                row_columns.append(column)
                break
            layer_index += 1

        self._layer_of_each_entry.append(layer_index)
        self._merkle_leaves.append(_hash(value.encode("utf-8")))
        return position

    def search(self, value: str) -> Optional[int]:
        """
        Searches for 'value' in the index and returns the position of the
        first occurrence found (scanning the epochs from the oldest to the
        most recent, and for each one the layers from the lowest to the
        highest), or None if it is not present in any epoch.

        For every existing epoch, and for each layer:
          1. the row in which 'value' would have been recorded in that
             epoch, at that layer, is computed. Thanks to the grouping (see
             _row), consecutive epochs of the same group, at the same
             layer, produce the same row: here it is recomputed at every
             iteration for simplicity of the code, but a real client would
             exploit this stability to retrieve the data of a whole group
             of epochs in a single access, instead of repeating the
             operation one by one;
          2. if that row is empty, we move on immediately to the next layer
             (or to the next epoch, if the layers are over): the bulk of
             the saving compared with a full scan of the log is that few
             rows are read per epoch, not the whole map nor the whole log;
          3. otherwise all the possible positions of that epoch are tried,
             one by one: for each one the column that 'value' would have if
             it were really at that position is computed (_column depends
             on the position, not on the layer), and we check whether it
             appears among the columns recorded in the row;
          4. a matching column is only a candidate (two different values
             may produce the same column by coincidence): it has to be
             confirmed by reading the real entry at that position. The
             position is returned only if it matches exactly.
        """
        for epoch in range(len(self.maps)):
            for layer_index in range(len(self.mapping_frequency_by_layer)):
                row = self._row(value, epoch, layer_index)
                row_columns = self.maps[epoch][row]

                if len(row_columns) == 0:
                    continue

                epoch_start = epoch * self.entries_per_epoch
                epoch_end = min(epoch_start + self.entries_per_epoch, len(self.entries))

                for candidate_position in range(epoch_start, epoch_end):
                    expected_column = self._column(value, candidate_position)
                    if expected_column in row_columns:
                        if self.entries[candidate_position] == value:
                            return candidate_position

        return None

    def root(self) -> bytes:
        """Cryptographic commitment (Merkle root) over the whole content of
        the log, in the order in which the entries were inserted."""
        return merkle_root(self._merkle_leaves)


def main() -> None:
    string_array = [
        "apple", "banana", "orange", "kiwi", "pear", "grape", "strawberry",
        "pineapple", "mango", "papaya", "cherry", "lemon", "peach",
        "apricot", "melon", "watermelon", "fig", "medlar", "persimmon",
        "blueberry"]

    #Initialization
    index = LogIndex(entries_per_epoch=5, number_of_rows=4, column_bits=6)


    print("\n=== Insertion of the string array ===")
    for word in string_array:
        position = index.insert(word)


    print(f"\nEntries inserted: {len(string_array)}")
    print(f"Epochs created so far: {len(index.maps)}")
    print(f"Merkle root of the LogIndex: {index.root().hex()}")

    print("\n=== The row stays stable for a group of epochs, then changes (layer 0) ===")
    # At layer 0, mapping_frequency_by_layer[0] = 4: epochs 0, 1, 2 and 3
    # share the same row computation (same group); epoch 4 opens a new
    # group and the row may change. We show it by computing the row that
    # "apple" would have in each of these epochs, WITHOUT reinserting it:
    # _row only depends on the value, the epoch and the layer.
    layer_0_frequency = index.mapping_frequency_by_layer[0]
    for test_epoch in range(0, layer_0_frequency + 1):
        computed_row = index._row("apple", test_epoch, layer_index=0)
        group = test_epoch // layer_0_frequency
        print(f"  'apple' in epoch {test_epoch} (group {group}) -> row {computed_row}")
    print(f"  The first {layer_0_frequency} epochs (same group) give the SAME row: a client can")
    print("read or prove (Merkle proof) the data of the whole group with a single access.")

    print("\n=== The same word, in different groups of epochs, may end up in different rows ===")
    # "banana" has already been inserted once (position 1, epoch 0). It is
    # reinserted here, many entries later: it falls in an epoch of a later
    # group (at layer 0), and its row is recomputed from scratch.
    first_position = 1
    first_epoch = first_position // index.entries_per_epoch
    first_row = index._row("banana", first_epoch, layer_index=0)

    second_position = index.insert("banana")
    second_epoch = second_position // index.entries_per_epoch
    second_row = index._row("banana", second_epoch, layer_index=0)

    print(
        f"  1st 'banana' -> position {first_position}, epoch {first_epoch}, "
        f"group {first_epoch // layer_0_frequency} (layer 0), row {first_row}"
    )
    print(
        f"  2nd 'banana' -> position {second_position}, epoch {second_epoch}, "
        f"group {second_epoch // layer_0_frequency} (layer 0), row {second_row}"
    )
    if first_row != second_row:
        print("  Different groups, DIFFERENT ROWS: the filter map does not always 'get dirty' in the same place.")
    else:
        print("  (in this run the two rows coincide by chance: it can happen, it is not guaranteed)")


    print("\n=== Search for values PRESENT in the array ===")
    for word in ["banana", "blueberry", "fig"]:
        position = index.search(word)
        print(
            f"  '{word}': LogIndex -> found at position {position}"
        )

    print("\n=== Search for values ABSENT from the array ===")
    # "almond" was never inserted, but its hash switches on by coincidence
    # all the bits already switched on by other words: it is a Bloom filter
    # false positive. "avocado" instead is not, and is correctly
    # recognised as absent by both structures.
    for word in ["avocado", "almond"]:
        position = index.search(word)
        print(
            f"  '{word}': LogIndex -> {position} (None = absent)"
        )

    # print("\n=== Print of the whole filter map ===")

    # for epoch in range(len(index.maps)):
    #     layer_0_group = epoch // layer_0_frequency
    #     print(f"\n  epoch {epoch} (layer 0 group: {layer_0_group}):")
    #     epoch_start = epoch * index.entries_per_epoch
    #     epoch_end = min(epoch_start + index.entries_per_epoch, len(index.entries))

    #     for row in range(index.number_of_rows):
    #         stored_columns = index.maps[epoch][row]

    #         entries_of_this_row = []
    #         for position in range(epoch_start, epoch_end):
    #             value = index.entries[position]
    #             real_layer = index._layer_of_each_entry[position]
    #             if index._row(value, epoch, real_layer) == row:
    #                 entries_of_this_row.append(f"{value} (layer {real_layer})")

    #         print(f"    row {row}: columns={stored_columns} -> entries={entries_of_this_row}")


if __name__ == "__main__":
    main()
