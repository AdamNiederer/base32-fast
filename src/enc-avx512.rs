use std::arch::x86_64::*;

use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z};
use super::{RFC4648_LUT, RFC4648HEX_LUT, CROCKFORD_LUT, GEOHASH_LUT, Z_LUT};

pub(crate) unsafe fn to_char_avx512<const A: u8>(src: __m512i) -> __m512i {
    let lut = match A {
        Rfc4648 => &RFC4648_LUT,
        Rfc4648Hex => &RFC4648HEX_LUT,
        Crockford => &CROCKFORD_LUT,
        Geohash => &GEOHASH_LUT,
        Z => &Z_LUT,
        _ => core::hint::unreachable_unchecked(),
    };

    let lut_reg = _mm512_loadu_si512(lut.as_ptr() as *const _);
    _mm512_permutexvar_epi8(src, lut_reg)
}

pub(crate) unsafe fn b32enc_avx512<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 40 {
        let s = _mm512_maskz_loadu_epi8(0x000000FFFFFFFFFF, src.as_ptr().add(src_cur) as *const i8);
        let shuf = _mm512_set_epi8(
            35, 36, 37, 38, 39, 39, 39, 39,
            30, 31, 32, 33, 34, 34, 34, 34,
            25, 26, 27, 28, 29, 29, 29, 29,
            20, 21, 22, 23, 24, 24, 24, 24,
            15, 16, 17, 18, 19, 19, 19, 19,
            10, 11, 12, 13, 14, 14, 14, 14,
            5, 6, 7, 8, 9, 9, 9, 9,
            0, 1, 2, 3, 4, 4, 4, 4,
        );
        let p = _mm512_permutexvar_epi8(shuf, s);
        let multishift = _mm512_set_epi8(
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
            24, 29, 34, 39, 44, 49, 54, 59,
        );
        let shifted = _mm512_multishift_epi64_epi8(multishift, p);
        let masked = _mm512_and_si512(shifted, _mm512_set1_epi8(0x1F));
        let res = to_char_avx512::<{A}>(masked);
        _mm512_storeu_si512(dst.as_ptr().add(dst_cur) as *mut __m512i, res);
        src_cur += 40;
        dst_cur += 64;
    }
    return dst;
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;
    use crate::{RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS};

    static TO_CHAR_INPUT: [u8; 64] = [
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ];

    #[test]
    fn test_rfc4648_avx512() {
        unsafe {
            let src = _mm512_loadu_si512(TO_CHAR_INPUT.as_ptr() as *const _);
            let result = to_char_avx512::<Rfc4648>(src);
            let mut actual = [0u8; 64];
            _mm512_storeu_si512(actual.as_mut_ptr() as *mut _, result);
            assert_eq!(&actual[0..32], RFC4648_CHARS);
        }
    }

    #[test]
    fn test_rfc4648hex_avx512() {
        unsafe {
            let src = _mm512_loadu_si512(TO_CHAR_INPUT.as_ptr() as *const _);
            let result = to_char_avx512::<Rfc4648Hex>(src);
            let mut actual = [0u8; 64];
            _mm512_storeu_si512(actual.as_mut_ptr() as *mut _, result);
            assert_eq!(&actual[0..32], RFC4648HEX_CHARS);
        }
    }

    #[test]
    fn test_crockford_avx512() {
        unsafe {
            let src = _mm512_loadu_si512(TO_CHAR_INPUT.as_ptr() as *const _);
            let result = to_char_avx512::<Crockford>(src);
            let mut actual = [0u8; 64];
            _mm512_storeu_si512(actual.as_mut_ptr() as *mut _, result);
            assert_eq!(&actual[0..32], CROCKFORD_CHARS);
        }
    }

    #[test]
    fn test_geohash_avx512() {
        unsafe {
            let src = _mm512_loadu_si512(TO_CHAR_INPUT.as_ptr() as *const _);
            let result = to_char_avx512::<Geohash>(src);
            let mut actual = [0u8; 64];
            _mm512_storeu_si512(actual.as_mut_ptr() as *mut _, result);
            assert_eq!(&actual[0..32], GEOHASH_CHARS);
        }
    }

    #[test]
    fn test_z_avx512() {
        unsafe {
            let src = _mm512_loadu_si512(TO_CHAR_INPUT.as_ptr() as *const _);
            let result = to_char_avx512::<Z>(src);
            let mut actual = [0u8; 64];
            _mm512_storeu_si512(actual.as_mut_ptr() as *mut _, result);
            assert_eq!(&actual[0..32], Z_CHARS);
        }
    }

    #[bench]
    fn bench_to_char_avx512(b: &mut Bencher) {
        let input = [0; 64];
        unsafe {
            let src_reg = _mm512_loadu_si512(input.as_ptr() as *const _);
            b.iter(|| {
                black_box(to_char_avx512::<Z>(black_box(src_reg)));
            });
        }
    }

    #[bench]
    fn bench_b32enc_avx512(b: &mut Bencher) {
        let input = [0; 40];
        let mut output = [0u8; 64];
        b.iter(|| {
            unsafe { black_box(b32enc_avx512::<Z>(black_box(&input), black_box(&mut output))) };
        });
    }
}
