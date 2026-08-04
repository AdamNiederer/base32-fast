# base32-fast

An adaptation of Wojciech Mula's AVX-512 base64 encoding/decoding
algorithms. Supports all of the major alphabets, and a portable simd path.

## Usage

Comes as a CLI tool and a library:

```
Usage: base32-fast [OPTIONS] [INPUT]

Arguments:
  [INPUT]  The file from which input will be read. If not provided, read from stdin

Options:
  -d, --decode               Whether to decode the input
  -e, --encode               Whether to encode the input
      --alphabet <ALPHABET>  The alphabet to use (rfc4648, rfc4648hex, crockford, geohash, or z; default rfc4648)
  -n                         Whether to not output a newline after the encoded or decoded text
  -h, --help                 Print help
```

```rust
use base32_fast::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z, b32enc, b32dec};

let encoded = b32enc(&src, &mut dst, Geohash);
let decoded = b32dec(&src, &mut dst, Geohash);
```

## Performance

`base32-fast` provides three code paths: A reference scalar implementation, an
implementation using Rust's `std::simd`, and a handwritten AVX-512
implementation.

If you can use the AVX-512 path, do so. The encoder uses multishift instructions
and the decoder abuses integer FMA instructions to maximize speed, and the SIMD
implementation can do neither.

`-Zbuild-std` is required for the portable SIMD path to be fast on some
architectures, especially those with AVX-512.

Benchmarks of each code path per 64-byte block:

```
test dec::tests::bench_b32dec           ... bench:          21.84 ns/iter (+/- 0.59)
test dec::tests::bench_b32dec_simd      ... bench:           2.76 ns/iter (+/- 0.01)
test dec::tests::bench_b32dec_avx512    ... bench:           1.16 ns/iter (+/- 0.17)
test enc::tests::bench_b32enc           ... bench:          58.35 ns/iter (+/- 0.42)
test enc::tests::bench_b32enc_simd      ... bench:          10.42 ns/iter (+/- 0.02)
test enc::tests::bench_b32enc_avx512    ... bench:           1.14 ns/iter (+/- 0.03)
```

Equivalent benchmarks of the `base32` crate for comparison:

```
test dec::tests::bench_base32_decode    ... bench:          39.12 ns/iter (+/- 0.34)
test enc::tests::bench_base32_encode    ... bench:          66.15 ns/iter (+/- 1.57)
```

Benchmarks with 1MB of data:

```
test dec::tests::bench_b32dec_avx512_bulk ... bench:     199,036.20 ns/iter (+/- 20,178.75)
test dec::tests::bench_b32dec_simd_bulk   ... bench:     661,083.40 ns/iter (+/- 4,813.39)
test enc::tests::bench_b32enc_avx512_bulk ... bench:     184,308.08 ns/iter (+/- 2,036.40)
test enc::tests::bench_b32enc_simd_bulk   ... bench:     351,228.25 ns/iter (+/- 4,039.00)
```

Compared to base32:

```
test dec::tests::bench_base32_decode_bulk ... bench:   8,315,518.80 ns/iter (+/- 77,973.30)
test enc::tests::bench_base32_encode_bulk ... bench:   7,334,881.90 ns/iter (+/- 106,046.64)
```
