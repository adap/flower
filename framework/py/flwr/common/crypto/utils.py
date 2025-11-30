import sys
import math
import pickle  # solo per stimare la size serializzata in memoria

def log_serialization_size(obj, tag: str, mtu: int = 1500, header_overhead: int = 40):
    """
    Stima la dimensione serializzata e il numero di pacchetti IP attesi.

    Args:
        obj: oggetto da serializzare (es. RecordDict).
        tag: stringa descrittiva (es. 'fitres', 'fitins').
        mtu: MTU della rete in byte (default 1500).
        header_overhead: tipico overhead IP+TCP (40 byte senza opzioni).
    """
    try:
        # serializza in binario (pickle è solo per stimare la size in memoria)
        data = pickle.dumps(obj)
        size_bytes = len(data)

        payload_per_packet = mtu - header_overhead
        n_packets = math.ceil(size_bytes / payload_per_packet)

        #print(f"[NET-DEBUG] {tag}: {size_bytes/1024:.2f} KB "
            #  f"→ ~{n_packets} pacchetti TCP/IP (MTU={mtu})")
    except Exception as e:
       print(f"[NET-DEBUG] {tag}: errore stima size - {e}", file=sys.stderr)
