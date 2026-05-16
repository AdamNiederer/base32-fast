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
to achieve 0.37 cycles per byte, and the decoder abuses integer FMA instructions
to achieve 0.4 cycles per byte on Zen 5.

`-Zbuild-std` is required for the portable SIMD path to be fast on some
architectures, especially those with AVX-512.

```
$ RUSTFLAGS="-C target-cpu=native" cargo bench --features simd,avx-512 -Zbuild-std
test dec::tests::bench_b32dec           ... bench:          22.10 ns/iter (+/- 0.75)
test dec::tests::bench_b32dec_avx512    ... bench:           2.86 ns/iter (+/- 0.00)
test dec::tests::bench_from_char        ... bench:          18.25 ns/iter (+/- 5.96)
test dec::tests::bench_from_char_avx512 ... bench:           0.43 ns/iter (+/- 0.00)
test dec::tests::bench_from_char_simd   ... bench:           0.43 ns/iter (+/- 0.01)
test dec::tests::bench_padcount         ... bench:           8.47 ns/iter (+/- 0.40)
test dec::tests::bench_padcount_avx512  ... bench:           2.57 ns/iter (+/- 0.00)
test enc::tests::bench_b32enc           ... bench:          36.37 ns/iter (+/- 0.07)
test enc::tests::bench_b32enc_avx512    ... bench:           6.28 ns/iter (+/- 0.01)
test enc::tests::bench_b32enc_simd      ... bench:          10.42 ns/iter (+/- 0.07)
test enc::tests::bench_to_char          ... bench:          18.99 ns/iter (+/- 5.20)
test enc::tests::bench_to_char_avx512   ... bench:           0.38 ns/iter (+/- 0.00)
```
