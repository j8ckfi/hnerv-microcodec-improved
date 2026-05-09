"""Self-describing archive container (v2) for the HNeRV ft microcodec.

Layout (all little-endian):

    "HNV2"                   4   magic
    version                  1   currently 1
    flags                    1   bit0 = has sidecar
    decoder_len              4
    latent_len               4
    sidecar_len              4
    sd_crc32                 4   CRC32 over decoded state_dict bytes
    decoder_blob             decoder_len bytes
    latent_blob              latent_len bytes
    sidecar_blob             sidecar_len bytes (may be 0)
    archive_crc32            4   CRC32 over everything above

Decoder blob (replaces v1's per-byte stream-boundary search):

    n_streams                1
    for each stream:
        comp_len             4   compressed Brotli stream length (u32 LE)
        brotli bytes         comp_len

Why v2:
- v1 hardcoded section offsets in code (DECODER_BLOB_LEN, LATENT_BLOB_LEN);
  any retrain shifted them and the format silently broke.
- v1 dispatched the sidecar variant by raw payload length, with six magic
  numbers that collide trivially with arbitrary brotli output.
- v1 found Brotli stream boundaries by feeding the decoder one byte at a
  time (~160k Python calls per inflate); we now prefix each stream length.
- v1 had no integrity check; a flipped bit silently produces garbage frames.

Fixed overhead vs v1: 22 byte header + 4 byte trailing CRC + 1 stream
count + 7 * 4 stream-length prefixes = 55 bytes (~0.031% on the 178 KB
archive). v2 keeps the same Brotli streams and same byte-level
encodings, so the compressed size is otherwise identical.
"""
import struct
import zlib

import numpy as np
import torch

import codec as v1


MAGIC = b"HNV2"
VERSION = 1
FLAG_HAS_SIDECAR = 0x1
HEADER_FMT = "<4sBBIII I"  # magic, ver, flags, dec_len, lat_len, sid_len, sd_crc
HEADER_LEN = struct.calcsize(HEADER_FMT)
TAIL_LEN = 4  # archive crc32


def _state_dict_fingerprint_bytes(state_dict):
    """Concatenate tensors in DECODER_STORAGE_ORDER for a stable CRC input."""
    probe = v1.HNeRVDecoder(
        latent_dim=v1.LATENT_DIM,
        base_channels=v1.BASE_CHANNELS,
        eval_size=v1.EVAL_SIZE,
    )
    items = list(probe.state_dict().keys())
    parts = []
    for idx in v1.DECODER_STORAGE_ORDER:
        name = items[idx]
        parts.append(state_dict[name].detach().cpu().numpy().tobytes())
    return b"".join(parts)


def encode_decoder_blob(streams):
    """Pack a list of pre-compressed Brotli stream bytes with length prefixes."""
    if len(streams) > 255:
        raise ValueError("too many decoder streams")
    out = [bytes([len(streams)])]
    for s in streams:
        out.append(struct.pack("<I", len(s)))
        out.append(s)
    return b"".join(out)


def decode_decoder_blob(data):
    n_streams = data[0]
    pos = 1
    streams = []
    for _ in range(n_streams):
        if pos + 4 > len(data):
            raise ValueError("truncated decoder stream length prefix")
        comp_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + comp_len > len(data):
            raise ValueError("truncated decoder stream body")
        streams.append(data[pos:pos + comp_len])
        pos += comp_len
    if pos != len(data):
        raise ValueError("trailing decoder bytes")
    return streams


def decode_decoder_from_streams(streams):
    """Same dequant pipeline as v1, but consumes pre-split streams directly."""
    import brotli
    raw = b"".join(brotli.decompress(s) for s in streams)
    probe = v1.HNeRVDecoder(
        latent_dim=v1.LATENT_DIM,
        base_channels=v1.BASE_CHANNELS,
        eval_size=v1.EVAL_SIZE,
    )
    items = list(probe.state_dict().items())
    pos = 0
    sd = {}
    for idx in v1.DECODER_STORAGE_ORDER:
        name, tensor = items[idx]
        shape = tuple(tensor.shape)
        numel = int(tensor.numel())
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        pos += 2
        q = v1.decode_mapped_u8(zz, v1.DECODER_BYTE_MAPS.get(idx, "zig"))
        if len(shape) == 4:
            storage_perm = v1.CONV4_STORAGE_PERMS[idx]
            inverse_perm = v1.CONV4_INVERSE_PERMS[idx]
            stored_shape = tuple(shape[i] for i in storage_perm)
            q = q.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q.reshape(shape)
        sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)
    if pos != len(raw):
        raise ValueError("trailing decoder payload after dequant")
    return sd


def encode_archive(decoder_streams, latent_blob, sidecar_blob, sd_crc):
    decoder_blob = encode_decoder_blob(decoder_streams)
    flags = FLAG_HAS_SIDECAR if sidecar_blob else 0
    sid_len = len(sidecar_blob) if sidecar_blob else 0
    header = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, flags,
        len(decoder_blob), len(latent_blob), sid_len,
        sd_crc & 0xFFFFFFFF,
    )
    body = header + decoder_blob + latent_blob + (sidecar_blob or b"")
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc)


def parse_archive(data):
    """Decode a v2 archive into (state_dict, latents, meta), like v1.parse_archive."""
    if len(data) < HEADER_LEN + TAIL_LEN:
        raise ValueError("archive too short")
    if data[:4] != MAGIC:
        raise ValueError("bad magic; not an HNV2 archive")
    body = data[:-TAIL_LEN]
    expected_crc = struct.unpack_from("<I", data, len(data) - TAIL_LEN)[0]
    actual_crc = zlib.crc32(body) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(f"archive CRC mismatch: {actual_crc:08x} != {expected_crc:08x}")

    (_, version, flags, dec_len, lat_len, sid_len, sd_crc) = struct.unpack_from(
        HEADER_FMT, data, 0
    )
    if version != VERSION:
        raise ValueError(f"unsupported archive version {version}")

    pos = HEADER_LEN
    decoder_blob = body[pos:pos + dec_len]; pos += dec_len
    latent_blob = body[pos:pos + lat_len]; pos += lat_len
    sidecar_blob = body[pos:pos + sid_len]; pos += sid_len
    if pos != len(body):
        raise ValueError("trailing bytes inside archive body")
    if bool(sid_len) != bool(flags & FLAG_HAS_SIDECAR):
        raise ValueError("sidecar flag/length disagreement")

    streams = decode_decoder_blob(decoder_blob)
    sd = decode_decoder_from_streams(streams)
    sd_actual = zlib.crc32(_state_dict_fingerprint_bytes(sd)) & 0xFFFFFFFF
    if sd_actual != sd_crc:
        raise ValueError(
            f"decoded state_dict CRC mismatch: {sd_actual:08x} != {sd_crc:08x}"
        )

    latents = v1.apply_latent_sidecar(
        v1.decode_latents_compact(latent_blob), sidecar_blob
    )
    meta = {
        "n_pairs": v1.N_PAIRS,
        "latent_dim": v1.LATENT_DIM,
        "base_channels": v1.BASE_CHANNELS,
        "eval_size": list(v1.EVAL_SIZE),
    }
    return sd, latents, meta


def split_v1_decoder_streams(v1_decoder_blob):
    """Recover the 7 individual Brotli streams from a v1 decoder blob.

    Mirrors v1.decompress_brotli_streams but only retains the compressed
    boundaries (it does not decompress). One-time cost, used by the
    v1->v2 re-encoder; not on the inflate path.
    """
    import brotli
    streams = []
    pos = 0
    for _ in range(len(v1.DECODER_STREAM_ENDS)):
        dec = brotli.Decompressor()
        start = pos
        while pos < len(v1_decoder_blob) and not dec.is_finished():
            dec.process(v1_decoder_blob[pos:pos + 1])
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated v1 decoder blob")
        streams.append(v1_decoder_blob[start:pos])
    if pos != len(v1_decoder_blob):
        raise ValueError("trailing v1 decoder bytes")
    return streams


def reencode_v1_to_v2(v1_bytes):
    """Convert a v1 archive blob to a v2 archive blob without retraining."""
    decoder_blob = v1_bytes[:v1.DECODER_BLOB_LEN]
    latent_blob = v1_bytes[
        v1.DECODER_BLOB_LEN:v1.DECODER_BLOB_LEN + v1.LATENT_BLOB_LEN
    ]
    sidecar_blob = v1_bytes[v1.DECODER_BLOB_LEN + v1.LATENT_BLOB_LEN:]

    streams = split_v1_decoder_streams(decoder_blob)
    sd, _, _ = v1.parse_archive(v1_bytes)
    sd_crc = zlib.crc32(_state_dict_fingerprint_bytes(sd)) & 0xFFFFFFFF
    return encode_archive(streams, latent_blob, sidecar_blob, sd_crc)
