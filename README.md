# hnerv-microcodec-improved

Improvements to commaai PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)
(`hnerv_ft_microcodec` submission). Two commits, separable:

1. **`cleanup: extract helpers, document invariants, no format change`**
   Pure refactor. `parse_archive` produces byte-identical output on the
   real `archive.zip` from the PR release.

2. **`format v2: self-describing container with CRC and length-prefixed streams`**
   New optional archive layout (`archive_v2.py`). Wraps the same byte-level
   encodings v1 already used, but adds magic + version + per-section
   lengths + CRC32 + per-stream length prefixes. Existing v1 archives
   keep working through `codec.parse_archive`; new archives can use
   `archive_v2.parse_archive`.

   Verified against the real `archive.zip`:

   ```
   v1 archive:           178158 bytes
   v2 archive:           178213 bytes  (+55, +0.031%)
   v1 decode fingerprint: 9cff13f6...849f319f
   v2 decode fingerprint: 9cff13f6...849f319f   (match)
   v1 parse_archive:     192.2 ms
   v2 parse_archive:      22.5 ms  (8.52x faster)
   ```

   The speedup comes from replacing v1's byte-at-a-time
   `brotli.Decompressor.process` loop (used to find stream boundaries)
   with explicit per-stream length prefixes.

   Integrity checks:
   - truncation rejected by archive CRC
   - single-bit flip in payload rejected by archive CRC
   - bad magic rejected with clean error
   - decoded `state_dict` rejected by independent `sd_crc32` if the schema
     drifts between encoder and decoder

## Run the test

```
cd submissions/hnerv_ft_microcodec
# place data/archive.zip from the PR release here, then:
unzip data/archive.zip -d data/
python tests/test_archive_v2.py
```

## What I did NOT do

- Retrain or re-fine-tune the HNeRV decoder.
- Change any score-affecting math (postprocess, sidecar deltas, latent
  dequant). Both commits are decode-byte-preserving against the released
  archive.
- Re-encode `archive.zip` for upstream submission. The v2 path here
  re-wraps the existing v1 payload; an actual submission using v2 would
  need the encoder side to produce the streams from real weights.
