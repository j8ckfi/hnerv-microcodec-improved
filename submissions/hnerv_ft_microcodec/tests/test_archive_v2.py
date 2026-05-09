"""Roundtrip tests against the real archive.zip from the PR release.

Run from submissions/hnerv_ft_microcodec/:
    python tests/test_archive_v2.py

Requires data/x to be unzipped from data/archive.zip.
"""
import hashlib
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import codec as v1            # noqa: E402
import archive_v2             # noqa: E402


def fingerprint(sd, latents):
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].numpy().tobytes())
    h.update(latents.numpy().tobytes())
    return h.hexdigest()


def main():
    v1_bytes = (ROOT / "data" / "x").read_bytes()
    print(f"v1 archive: {len(v1_bytes)} bytes")

    sd1, lat1, _ = v1.parse_archive(v1_bytes)
    fp1 = fingerprint(sd1, lat1)
    print(f"v1 decode fingerprint: {fp1}")

    v2_bytes = archive_v2.reencode_v1_to_v2(v1_bytes)
    overhead = len(v2_bytes) - len(v1_bytes)
    print(f"v2 archive: {len(v2_bytes)} bytes  ({overhead:+d} vs v1)")

    sd2, lat2, _ = archive_v2.parse_archive(v2_bytes)
    fp2 = fingerprint(sd2, lat2)
    print(f"v2 decode fingerprint: {fp2}")
    assert fp1 == fp2, "v2 roundtrip changed decoded output"

    # Truncation must fail cleanly.
    try:
        archive_v2.parse_archive(v2_bytes[:-1])
    except ValueError as e:
        print(f"truncation rejected: {e}")
    else:
        raise AssertionError("expected truncation to fail")

    # Bit-flip in payload must be caught by archive CRC.
    flipped = bytearray(v2_bytes)
    flipped[archive_v2.HEADER_LEN + 16] ^= 0x01
    try:
        archive_v2.parse_archive(bytes(flipped))
    except ValueError as e:
        msg = str(e)
        assert "CRC" in msg, f"unexpected error on bit flip: {msg}"
        print(f"bit flip rejected: {e}")
    else:
        raise AssertionError("expected bit flip to fail CRC")

    # Bad magic.
    try:
        archive_v2.parse_archive(b"XXXX" + v2_bytes[4:])
    except ValueError as e:
        print(f"bad magic rejected: {e}")
    else:
        raise AssertionError("expected bad magic to fail")

    print("OK: all roundtrip and integrity checks passed")


if __name__ == "__main__":
    main()
