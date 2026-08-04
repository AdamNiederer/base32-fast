use std::mem::transmute;
use std::simd::{Simd, Mask, Select};
use std::simd::cmp::SimdPartialOrd;

use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z};
use super::{CROCKFORD_LUT, GEOHASH_LUT, Z_LUT};

pub(crate) fn to_char_simd<const A: u8>(src: Simd<u8, 64>) -> Simd<u8, 64> {
    match A {
        Rfc4648 => {
            let off_a = Simd::splat(b'A');
            let off_2 = Simd::splat(b'2' - 26);
            let is_char_range: Mask<_, 64> = src.simd_lt(Simd::splat(26u8));
            let base_offset = is_char_range.select(off_a, off_2);
            src + base_offset
        }
        Rfc4648Hex => {
            let off_0 = Simd::splat(b'0');
            let off_a = Simd::splat(b'A' - 10);
            let is_digit_range: Mask<_, 64> = src.simd_lt(Simd::splat(10u8));
            let base_offset = is_digit_range.select(off_0, off_a);
            src + base_offset
        }
        Crockford => {
            let lut_reg = Simd::from_slice(&CROCKFORD_LUT);
            lut_reg.swizzle_dyn(src)
        }
        Geohash => {
            let lut_reg = Simd::from_slice(&GEOHASH_LUT);
            lut_reg.swizzle_dyn(src)
        }
        Z => {
            let lut_reg = Simd::from_slice(&Z_LUT);
            lut_reg.swizzle_dyn(src)
        }
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

pub(crate) fn b32enc_simd<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    const shuf: Simd<u8, 64> = Simd::from_array([
        4, 4, 4, 4, 3, 2, 1, 0,
        9, 9, 9, 9, 8, 7, 6, 5,
        14, 14, 14, 14, 13, 12, 11, 10,
        19, 19, 19, 19, 18, 17, 16, 15,
        24, 24, 24, 24, 23, 22, 21, 20,
        29, 29, 29, 29, 28, 27, 26, 25,
        34, 34, 34, 34, 33, 32, 31, 30,
        39, 39, 39, 39, 38, 37, 36, 35,
    ]);

    const endian64: Simd<u8, 64> = Simd::from_array([
        7, 6, 5, 4, 3, 2, 1, 0,
        15, 14, 13, 12, 11, 10, 9, 8,
        23, 22, 21, 20, 19, 18, 17, 16,
        31, 30, 29, 28, 27, 26, 25, 24,
        39, 38, 37, 36, 35, 34, 33, 32,
        47, 46, 45, 44, 43, 42, 41, 40,
        55, 54, 53, 52, 51, 50, 49, 48,
        63, 62, 61, 60, 59, 58, 57, 56
    ]);

    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 40 {
        let s = unsafe { transmute::<_, *const Simd<u8, 64>>(src.as_ptr().add(src_cur)).read_unaligned() };
        let p = unsafe { transmute::<_, Simd<u64, 8>>(s.swizzle_dyn(shuf)) };
        let d = (p >> Simd::splat(3)) & Simd::splat(0x1F00000000000000)
            | (p >> Simd::splat(6)) & Simd::splat(0x001F000000000000)
            | (p >> Simd::splat(9)) & Simd::splat(0x00001F0000000000)
            | (p >> Simd::splat(12)) & Simd::splat(0x0000001F00000000)
            | (p >> Simd::splat(15)) & Simd::splat(0x000000001F000000)
            | (p >> Simd::splat(18)) & Simd::splat(0x00000000001F0000)
            | (p >> Simd::splat(21)) & Simd::splat(0x0000000000001F00)
            | (p >> Simd::splat(24)) & Simd::splat(0x000000000000001F);

        let db = unsafe { transmute::<_, Simd<u8, 64>>(d).swizzle_dyn(endian64) };
        let res: Simd<u8, 64> = to_char_simd::<A>(db);

        unsafe { transmute::<_, *mut Simd<u8, 64>>(dst.as_ptr().add(dst_cur)).write_unaligned(res) };
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
    fn test_rfc4648_simd() {
        let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
        let result_simd = to_char_simd::<Rfc4648>(src);
        let mut actual = [0u8; 64];
        result_simd.copy_to_slice(&mut actual);
        assert_eq!(&actual[0..32], RFC4648_CHARS);
    }

    #[test]
    fn test_rfc4648hex_simd() {
        let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
        let result_simd = to_char_simd::<Rfc4648Hex>(src);
        let mut actual = [0u8; 64];
        result_simd.copy_to_slice(&mut actual);
        assert_eq!(&actual[0..32], RFC4648HEX_CHARS);
    }

    #[test]
    fn test_crockford_simd() {
        let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
        let result_simd = to_char_simd::<Crockford>(src);
        let mut actual = [0u8; 64];
        result_simd.copy_to_slice(&mut actual);
        assert_eq!(&actual[0..32], CROCKFORD_CHARS);
    }

    #[test]
    fn test_geohash_simd() {
        let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
        let result_simd = to_char_simd::<Geohash>(src);
        let mut actual = [0u8; 64];
        result_simd.copy_to_slice(&mut actual);
        assert_eq!(&actual[0..32], GEOHASH_CHARS);
    }

    #[test]
    fn test_z_simd() {
        let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
        let result = to_char_simd::<Z>(src);
        let mut actual = [0u8; 64];
        result.copy_to_slice(&mut actual);
        assert_eq!(&actual[0..32], Z_CHARS);
    }

    #[bench]
    fn bench_b32enc_simd(b: &mut Bencher) {
        let input = [0; 40];
        let mut output = [0u8; 64];
        b.iter(|| {
            black_box(b32enc_simd::<{Z}>(black_box(&input), black_box(&mut output)));
        });
    }

    #[bench]
    fn bench_b32enc_simd_bulk(b: &mut Bencher) {
        let input = vec![0u8; 10485760];
        let mut output = vec![0u8; 16777216];
        b.iter(|| {
            black_box(b32enc_simd::<Z>(black_box(&input), black_box(&mut output)));
        });
    }
}
