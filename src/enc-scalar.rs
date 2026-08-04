#[cfg(feature = "avx-512")]
use super::enc_avx512::b32enc_avx512;
#[cfg(all(feature = "simd", not(feature = "avx-512")))]
use super::enc_simd::b32enc_simd;

pub(crate) fn b32enc_generic<const A: u8>(src: &[u8], dst: &mut [u8]) {
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let simd_src_len = (src.len() / 40) * 40;
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let simd_dst_len = (simd_src_len / 40) * 64;
    #[cfg(not(any(feature = "avx-512", feature = "simd")))]
    let simd_src_len = 0;
    #[cfg(not(any(feature = "avx-512", feature = "simd")))]
    let simd_dst_len = 0;
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    if simd_src_len > 0 {
        #[cfg(feature = "avx-512")]
        b32enc_avx512::<A>(&src[..simd_src_len], &mut dst[..simd_dst_len]);
        #[cfg(all(feature = "simd", not(feature = "avx-512")))]
        b32enc_simd::<A>(&src[..simd_src_len], &mut dst[..simd_dst_len]);
    }

    let rem_src = &src[simd_src_len..];
    let rem_dst = &mut dst[simd_dst_len..];
    let mut rem_dst_cur = 0;
    for src_chunk in rem_src.chunks(5) {
        let dst_chunk = &mut rem_dst[rem_dst_cur..];
        let mut padded_chunk = [0u8; 5];
        padded_chunk[..src_chunk.len()].copy_from_slice(src_chunk);

        dst_chunk[0] = super::to_char::<A>((padded_chunk[0] & 0xf8) >> 3);
        dst_chunk[1] = super::to_char::<A>(((padded_chunk[0] & 0x07) << 2) | ((padded_chunk[1] & 0xC0) >> 6));
        dst_chunk[2] = super::to_char::<A>((padded_chunk[1] & 0x3E) >> 1);
        dst_chunk[3] = super::to_char::<A>(((padded_chunk[1] & 0x01) << 4) | ((padded_chunk[2] & 0xF0) >> 4));
        dst_chunk[4] = super::to_char::<A>(((padded_chunk[2] & 0x0F) << 1) | (padded_chunk[3] >> 7));
        dst_chunk[5] = super::to_char::<A>((padded_chunk[3] & 0x7C) >> 2);
        dst_chunk[6] = super::to_char::<A>(((padded_chunk[3] & 0x03) << 3) | ((padded_chunk[4] & 0xE0) >> 5));
        dst_chunk[7] = super::to_char::<A>(padded_chunk[4] & 0x1F);

        let dst_len = (src_chunk.len() * 8).div_ceil(5);
        for i in dst_len..8 {
            dst_chunk[i] = b'=';
        }

        rem_dst_cur += 8;
    }
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;
    use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z, RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS};

    #[test]
    fn test_to_char_scalar_rfc4648() {
        for value in 0..32u8 {
            let expected = RFC4648_CHARS[value as usize];
            let actual = super::super::to_char::<{Rfc4648}>(value);
            assert_eq!(actual, expected, "Rfc4648 mismatch for value {}", value);
        }
    }

    #[test]
    fn test_to_char_scalar_rfc4648hex() {
        for value in 0..32u8 {
            let expected = RFC4648HEX_CHARS[value as usize];
            let actual = super::super::to_char::<{Rfc4648Hex}>(value);
            assert_eq!(actual, expected, "Rfc4648Hex mismatch for value {}", value);
        }
    }

    #[test]
    fn test_to_char_scalar_crockford() {
        for value in 0..32u8 {
            let expected = CROCKFORD_CHARS[value as usize];
            let actual = super::super::to_char::<{Crockford}>(value);
            assert_eq!(actual, expected, "Crockford mismatch for value {}", value);
        }
    }

    #[test]
    fn test_to_char_scalar_geohash() {
        for value in 0..32u8 {
            let expected = GEOHASH_CHARS[value as usize];
            let actual = super::super::to_char::<{Geohash}>(value);
            assert_eq!(actual, expected, "Geohash mismatch for value {}", value);
        }
    }

    #[test]
    fn test_to_char_scalar_z() {
        for value in 0..32u8 {
            let expected = Z_CHARS[value as usize];
            let actual = super::super::to_char::<{Z}>(value);
            assert_eq!(actual, expected, "Z mismatch for value {}", value);
        }
    }

    #[bench]
    fn bench_to_char(b: &mut Bencher) {
        let input: Vec<u8> = (0..32).chain(0..32).collect();
        b.iter(|| {
            for src in input.iter() {
                black_box(super::super::to_char::<{Z}>(black_box(*src)));
            }
        });
    }

    #[bench]
    fn bench_b32enc(b: &mut Bencher) {
        let input = [0; 35];
        let mut output = [0u8; 56];
        b.iter(|| {
            b32enc_generic::<{Z}>(black_box(&input), black_box(&mut output));
        });
    }

    #[bench]
    fn bench_b32enc_bulk(b: &mut Bencher) {
        let input = vec![0u8; 10485760];
        let mut output = vec![0u8; 16777216];
        b.iter(|| {
            b32enc_generic::<Z>(black_box(&input), black_box(&mut output));
        });
    }
}
