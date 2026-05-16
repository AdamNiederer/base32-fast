#[cfg(feature = "avx-512")]
use super::dec_avx512::{padcount_avx512, b32dec_avx512};
#[cfg(all(feature = "simd", not(feature = "avx-512")))]
use super::dec_simd::b32dec_simd;

pub(crate) unsafe fn b32dec_generic<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    if src.len() >= 64 {
        #[cfg(feature = "avx-512")]
        b32dec_avx512::<A>(src, dst);
        #[cfg(all(feature = "simd", not(feature = "avx-512")))]
        b32dec_simd::<A>(src, dst);
    }

    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let src_tail = src.len() - src.len() % 64;
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let dst_tail = src_tail / 64 * 40;

    #[cfg(not(any(feature = "avx-512", feature = "simd")))]
    let src_tail = 0;
    #[cfg(not(any(feature = "avx-512", feature = "simd")))]
    let dst_tail = 0;

    let pad_count = if src.len() % 8 == 0 {
        #[cfg(feature = "avx-512")] {
            padcount_avx512(&src[src.len() - 8..])
        }
        #[cfg(not(feature = "avx-512"))] {
            padcount(&src[src.len() - 8..])
        }
    } else {
        8 - (src.len() % 8)
    };

    for (i, src_chunk) in src[src_tail..].chunks(8).enumerate() {
        let dst_chunk = &mut dst[(dst_tail + 5 * i)..];
        let mut padded_chunk = [b'='; 8];
        padded_chunk[..src_chunk.len()].copy_from_slice(src_chunk);

        let data0 = super::from_char::<A>(padded_chunk[0]);
        let data1 = super::from_char::<A>(padded_chunk[1]);
        let data2 = super::from_char::<A>(padded_chunk[2]);
        let data3 = super::from_char::<A>(padded_chunk[3]);
        let data4 = super::from_char::<A>(padded_chunk[4]);
        let data5 = super::from_char::<A>(padded_chunk[5]);
        let data6 = super::from_char::<A>(padded_chunk[6]);
        let data7 = super::from_char::<A>(padded_chunk[7]);

        dst_chunk[0] = (data0 << 3) | (data1 >> 2);
        dst_chunk[1] = (data1 << 6) | (data2 << 1) | (data3 >> 4);
        dst_chunk[2] = (data3 << 4) | (data4 >> 1);
        dst_chunk[3] = (data4 << 7) | (data5 << 2) | (data6 >> 3);
        dst_chunk[4] = (data6 << 5) | data7;
    }

    return &dst[..dst.len() - 5 + (8 - pad_count) * 5 / 8]
}

#[inline(always)]
pub fn padcount(src: &[u8]) -> usize {
    debug_assert_eq!(src.len(), 8, "Input slice must be exactly 8 bytes long");
    let mut count = 0;
    for i in (0..8).rev() {
        if src[i] == b'=' {
            count += 1;
        } else {
            break;
        }
    }
    count
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;
    use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z, RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS};

    fn expected_from_char(src: u8, alphabet: &[u8; 32]) -> u8 {
        if src == b'=' {
            return u8::MIN;
        }

        let lower = src.to_ascii_lowercase();
        let upper = src.to_ascii_uppercase();

        for (i, chr) in alphabet.iter().enumerate() {
            if lower == chr.to_ascii_lowercase() || upper == chr.to_ascii_uppercase() {
                return i as u8;
            }
        }

        u8::MAX
    }

    #[test]
    fn test_from_char_scalar_rfc4648() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, RFC4648_CHARS);
            unsafe {
                let actual = super::super::from_char::<Rfc4648>(value);
                assert_eq!(actual, expected, "Rfc4648 from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_rfc4648hex() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, RFC4648HEX_CHARS);
            unsafe {
                let actual = super::super::from_char::<Rfc4648Hex>(value);
                assert_eq!(actual, expected, "Rfc4648Hex from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_crockford() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, CROCKFORD_CHARS);
            unsafe {
                let actual = super::super::from_char::<Crockford>(value);
                assert_eq!(actual, expected, "Crockford from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_geohash() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, GEOHASH_CHARS);
            unsafe {
                let actual = super::super::from_char::<Geohash>(value);
                assert_eq!(actual, expected, "Geohash from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_z() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, Z_CHARS);
            unsafe {
                let actual = super::super::from_char::<Z>(value);
                assert_eq!(actual, expected, "Z from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_padcount_none() {
        let src: [u8; 8] = *b"ABCDEFGH";
        let count = padcount(&src);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_padcount_one() {
        let src: [u8; 8] = *b"ABCDEFG=";
        let count = padcount(&src);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_padcount_two() {
        let src: [u8; 8] = *b"ABCDEF==";
        let count = padcount(&src);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_padcount_all() {
        let src: [u8; 8] = *b"========";
        let count = padcount(&src);
        assert_eq!(count, 8);
    }

    #[bench]
    fn bench_from_char(b: &mut Bencher) {
        static FROM_CHAR_INPUT: [u8; 64] = [
            b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9',
            b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z',
            b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
            0, u8::MAX,
        ];
        unsafe {
            b.iter(|| {
                for src in FROM_CHAR_INPUT.iter() {
                    black_box(super::super::from_char::<{Z}>(black_box(*src)));
                }
            });
        }
    }

    static PADCOUNT_INPUT: [[u8; 8]; 9] = [
        *b"abcdefgh",
        *b"abcdefg=",
        *b"abcdef==",
        *b"abcde===",
        *b"abcd====",
        *b"abc=====",
        *b"ab======",
        *b"a=======",
        *b"========"
    ];

    #[bench]
    fn bench_padcount(b: &mut Bencher) {
        b.iter(|| {
            for input in PADCOUNT_INPUT.iter() {
                black_box(padcount(black_box(input)));
            }
        });
    }

    #[bench]
    fn bench_b32dec(b: &mut Bencher) {
        let input = b"GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBV";
        let mut output = [0u8; 35];
        b.iter(|| {
            unsafe { black_box(b32dec_generic::<{Z}>(black_box(input), black_box(&mut output))) };
        });
    }
}
