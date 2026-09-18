import hashlib
from typing import List, Optional


def _hash(dato: bytes) -> bytes: return hashlib.sha256(dato).digest()


def radice_merkle(foglie: List[bytes]) -> bytes:
    if not foglie: return _hash(b"")

    livello = list(foglie)
    while len(livello) > 1:
        if len(livello) % 2 == 1:
            livello.append(livello[-1])  # duplica l'ultimo nodo dispari

        livello_successivo = []
        for i in range(0, len(livello), 2):
            livello_successivo.append(_hash(livello[i] + livello[i + 1]))
        livello = livello_successivo

    return livello[0]


class LogIndex:
    """

    Componenti:
      - self.voci: il log sequenziale vero e proprio (equivalente alle
        "index entries" della EIP): ogni stringa inserita viene aggiunta in
        coda e la sua posizione (l'indice nell'array) diventa il suo
        identificativo permanente.
      - self.voci_per_epoca: quante voci consecutive condividono la stessa
        filter-map (equivalente a VALUES_PER_MAP nella EIP reale). Nella
        realtà è un numero grande (decine di migliaia); qui è piccolo per
        rendere visibile il passaggio da un'epoca alla successiva.
      - self.mapping_frequency_by_layer: una frequenza di raggruppamento
        delle epoche per ciascun livello (equivalente a
        2 ** LOG2_MAPPING_FREQUENCY nella EIP reale, dove vale
        [1024, 64, 4, 1]). Più alta è la frequenza di un livello, più
        epoche consecutive condividono la stessa riga a quel livello.
      - self.max_row_length_by_layer: quante colonne può contenere al
        massimo una riga di ciascun livello prima di dover spostare le
        voci in eccesso al livello successivo (equivalente a
        MAX_ROW_LENGTH nella EIP reale, dove vale [8, 168, 2728, 10920]).
      - self.mappe: una filter-map per ogni epoca (equivalenti alle
        "filter map" della EIP). self.mappe[epoca][riga] è la lista delle
        COLONNE (solo interi, nessuna posizione) registrate in quella riga
        di quell'epoca: la struttura non memorizza mai esplicitamente
        "questa colonna appartiene a questa posizione", esattamente come la
        EIP reale. È compito della ricerca ricostruirlo provando le
        posizioni candidate (vedi cerca()).

    I valori usati qui per mapping_frequency_by_layer e
    max_row_length_by_layer sono molto più piccoli di quelli della EIP,
    per restare visibili con solo poche decine di voci di esempio:
    la formula e il meccanismo (più livelli, ciascuno con una propria
    frequenza di raggruppamento e una propria capacità, con spostamento al
    livello successivo quando una riga è piena) sono però identici.
    """

    def __init__(
        self,
        voci_per_epoca: int = 5,
        numero_righe: int = 4,
        bit_colonna: int = 6,
        mapping_frequency_by_layer: Optional[List[int]] = None,
        max_row_length_by_layer: Optional[List[int]] = None,
    ):
        self.voci: List[str] = []
        self.voci_per_epoca = voci_per_epoca
        self.numero_righe = numero_righe
        self.bit_colonna = bit_colonna

        if mapping_frequency_by_layer is None:
            mapping_frequency_by_layer = [4, 2, 1, 1]
        if max_row_length_by_layer is None:
            max_row_length_by_layer = [2, 4, 8, 16]
        self.mapping_frequency_by_layer = mapping_frequency_by_layer
        self.max_row_length_by_layer = max_row_length_by_layer

        # Una mappa per epoca; ogni mappa è una lista di 'numero_righe'
        # righe; ogni riga è una lista di colonne (interi). 
        self.mappe: List[List[List[int]]] = []
        self._foglie_merkle: List[bytes] = []
        # Solo a fini illustrativi (non parte del meccanismo): tiene traccia
        # del livello in cui è effettivamente finita ogni voce inserita,
        # utile per mostrare nella demo quando avviene uno spostamento di
        # livello per riga piena.
        self._livello_di_ogni_voce: List[int] = []

    def _riga(self, valore: str, epoca: int, layer_index: int) -> int:

        mapping_frequency = self.mapping_frequency_by_layer[layer_index]
        masked_epoch = epoca - (epoca % mapping_frequency)
        dato = (
            valore.encode("utf-8")
            + masked_epoch.to_bytes(4, byteorder="big")
            + layer_index.to_bytes(4, byteorder="big")
        )
        hash_intero = int.from_bytes(_hash(dato), byteorder="big")
        return hash_intero % self.numero_righe

    def _colonna(self, valore: str, posizione: int) -> int:

        dato = valore.encode("utf-8") + posizione.to_bytes(4, byteorder="big")
        hash_intero = int.from_bytes(_hash(dato), byteorder="big")
        return hash_intero % (1 << self.bit_colonna)

    def inserisci(self, valore: str) -> int:
        posizione = len(self.voci)
        self.voci.append(valore)

        epoca = posizione // self.voci_per_epoca
        if epoca == len(self.mappe):
            # Prima voce di una nuova epoca: si apre una nuova filter-map
            # fatta di 'numero_righe' righe inizialmente vuote.
            nuova_mappa = []
            for _ in range(self.numero_righe):
                nuova_mappa.append([])
            self.mappe.append(nuova_mappa)

        colonna = self._colonna(valore, posizione)

        layer_index = 0
        while True:
            riga = self._riga(valore, epoca, layer_index)
            colonne_della_riga = self.mappe[epoca][riga]
            ultimo_layer_disponibile = len(self.max_row_length_by_layer) - 1
            capacita_massima = self.max_row_length_by_layer[
                min(layer_index, ultimo_layer_disponibile)
            ]
            if len(colonne_della_riga) < capacita_massima:
                colonne_della_riga.append(colonna)
                break
            layer_index += 1

        self._livello_di_ogni_voce.append(layer_index)
        self._foglie_merkle.append(_hash(valore.encode("utf-8")))
        return posizione

    def cerca(self, valore: str) -> Optional[int]:
        """
        Cerca 'valore' nell'indice e restituisce la posizione della prima
        occorrenza trovata (scandendo le epoche dalla più vecchia alla più
        recente, e per ciascuna i livelli dal più basso al più alto),
        oppure None se non è presente in nessuna epoca.

        Per ogni epoca già esistente, e per ciascun livello:
          1. si calcola la riga in cui 'valore' sarebbe stato registrato
             in quell'epoca, a quel livello. Grazie al raggruppamento (vedi
             _riga), epoche consecutive dello stesso gruppo, allo stesso
             livello, producono la stess riga: qui la si ricalcola
             comunque a ogni iterazione per semplicità del codice, ma un
             client reale sfrutterebbe questa stabilità per recuperare in
             un solo accesso i dati di un intero gruppo di epoche, invece
             di ripetere l'operazione una per una;
          2. se quella riga è vuota, si passa subito al livello successivo
             (o all'epoca successiva, se i livelli sono finiti): il grosso
             del risparmio rispetto a uno scorrimento completo del log è
             che si leggono poche righe per epoca, non l'intera mappa né
             l'intero log;
          3. altrimenti si provano, una per una, tutte le posizioni
             possibili di quell'epoca: per ciascuna si calcola la colonna
             che 'valore' avrebbe se fosse davvero a quella posizione
             (_colonna dipende dalla posizione, non dal livello), e si
             controlla se compare fra le colonne registrate nella riga;
          4. una colonna coincidente è solo un candidato (due valori
             diversi possono produrre la stessa colonna per coincidenza):
             va confermato leggendo la voce reale a quella posizione. Solo
             se corrisponde esattamente si restituisce la posizione.
        """
        for epoca in range(len(self.mappe)):
            for layer_index in range(len(self.mapping_frequency_by_layer)):
                riga = self._riga(valore, epoca, layer_index)
                colonne_della_riga = self.mappe[epoca][riga]

                if len(colonne_della_riga) == 0:
                    continue

                inizio_epoca = epoca * self.voci_per_epoca
                fine_epoca = min(inizio_epoca + self.voci_per_epoca, len(self.voci))

                for posizione_candidata in range(inizio_epoca, fine_epoca):
                    colonna_attesa = self._colonna(valore, posizione_candidata)
                    if colonna_attesa in colonne_della_riga:
                        if self.voci[posizione_candidata] == valore:
                            return posizione_candidata

        return None

    def radice(self) -> bytes:
        """Impegno crittografico (radice di Merkle) sull'intero contenuto
        del log, nell'ordine in cui le voci sono state inserite."""
        return radice_merkle(self._foglie_merkle)


def main() -> None:
    array_di_stringhe = [
        "mela", "banana", "arancia", "kiwi", "pera", "uva", "fragola",
        "ananas", "mango", "papaya", "ciliegia", "limone", "pesca",
        "albicocca", "melone", "anguria", "fico", "nespola", "cachi",
        "mirtillo"]

    #Inizializzazione
    indice = LogIndex(voci_per_epoca=5, numero_righe=4, bit_colonna=6)


    print("\n=== Inserimento dell'array di stringhe ===")
    for parola in array_di_stringhe:
        posizione = indice.inserisci(parola)


    print(f"\nElementi inseriti: {len(array_di_stringhe)}")
    print(f"Epoche create finora: {len(indice.mappe)}")
    print(f"Radice di Merkle del LogIndex: {indice.radice().hex()}")

    print("\n=== La riga resta stabile per un gruppo di epoche, poi cambia (livello 0) ===")
    # Al livello 0, mapping_frequency_by_layer[0] = 4: le epoche 0, 1, 2 e 3
    # condividono lo stesso calcolo di riga (stesso gruppo); l'epoca 4 apre
    # un nuovo gruppo e la riga può cambiare. Lo mostriamo calcolando la
    # riga che "mela" avrebbe in ciascuna di queste epoche, SENZA bisogno di
    # reinserirla: _riga dipende solo dal valore, dall'epoca e dal livello.
    frequenza_livello_0 = indice.mapping_frequency_by_layer[0]
    for epoca_di_prova in range(0, frequenza_livello_0 + 1):
        riga_calcolata = indice._riga("mela", epoca_di_prova, layer_index=0)
        gruppo = epoca_di_prova // frequenza_livello_0
        print(f"  'mela' nell'epoca {epoca_di_prova} (gruppo {gruppo}) -> riga {riga_calcolata}")
    print(
        f"  Le prime {frequenza_livello_0} epoche (stesso gruppo) danno la STESSA riga: un client può"
    )
    print(
        "  leggere o dimostrare (Merkle proof) i dati di tutto il gruppo con un solo accesso."
    )

    print("\n=== La stessa parola, in gruppi di epoche diversi, può finire in righe diverse ===")
    # "banana" è già stata inserita una volta (posizione 1, epoca 0). La
    # reinseriamo qui, molte voci dopo: cade in un'epoca di un gruppo
    # successivo (a livello 0), e la sua riga viene ricalcolata da zero.
    prima_posizione = 1
    prima_epoca = prima_posizione // indice.voci_per_epoca
    prima_riga = indice._riga("banana", prima_epoca, layer_index=0)

    seconda_posizione = indice.inserisci("banana")
    seconda_epoca = seconda_posizione // indice.voci_per_epoca
    seconda_riga = indice._riga("banana", seconda_epoca, layer_index=0)

    print(
        f"  1a 'banana' -> posizione {prima_posizione}, epoca {prima_epoca}, "
        f"gruppo {prima_epoca // frequenza_livello_0} (livello 0), riga {prima_riga}"
    )
    print(
        f"  2a 'banana' -> posizione {seconda_posizione}, epoca {seconda_epoca}, "
        f"gruppo {seconda_epoca // frequenza_livello_0} (livello 0), riga {seconda_riga}"
    )
    if prima_riga != seconda_riga:
        print("  Gruppi diversi, RIGHE DIVERSE: la filter-map non si 'sporca' sempre nello stesso punto.")
    else:
        print("  (in questa esecuzione le due righe coincidono per coincidenza: può capitare, non è garantito)")


    print("\n=== Ricerca di valori PRESENTI nell'array ===")
    for parola in ["banana", "mirtillo", "fico"]:
        posizione = indice.cerca(parola)
        print(
            f"  '{parola}': LogIndex -> trovato in posizione {posizione}"
        )

    print("\n=== Ricerca di valori ASSENTI dall'array ===")
    # "mandorla" non è mai stata inserita, ma il suo hash accende per
    # coincidenza tutti i bit già accesi da altre parole: è un falso
    # positivo del Bloom filter. "avocado" invece non lo è, ed è
    # correttamente riconosciuta come assente da entrambe le strutture.
    for parola in ["avocado", "mandorla"]:
        posizione = indice.cerca(parola)
        print(
            f"  '{parola}': LogIndex -> {posizione} (None = assente)"
        )

    # print("\n=== Stampa dell'intera filter map ===")

    # for epoca in range(len(indice.mappe)):
    #     gruppo_livello_0 = epoca // frequenza_livello_0
    #     print(f"\n  epoca {epoca} (gruppo di livello 0: {gruppo_livello_0}):")
    #     inizio_epoca = epoca * indice.voci_per_epoca
    #     fine_epoca = min(inizio_epoca + indice.voci_per_epoca, len(indice.voci))

    #     for riga in range(indice.numero_righe):
    #         colonne_memorizzate = indice.mappe[epoca][riga]

    #         voci_di_questa_riga = []
    #         for posizione in range(inizio_epoca, fine_epoca):
    #             valore = indice.voci[posizione]
    #             livello_reale = indice._livello_di_ogni_voce[posizione]
    #             if indice._riga(valore, epoca, livello_reale) == riga:
    #                 voci_di_questa_riga.append(f"{valore} (livello {livello_reale})")

    #         print(f"    riga {riga}: colonne={colonne_memorizzate} -> voci={voci_di_questa_riga}")


if __name__ == "__main__":
    main()
