use std::arch::x86_64::*;

use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z};

#[inline(always)]
pub(crate) unsafe fn from_char_avx512<const A: u8>(src: __m512i) -> __m512i {
    let lut = match A {
        Rfc4648 => &crate::dec::RFC4648_LUT,
        Rfc4648Hex => &crate::dec::RFC4648HEX_LUT,
        Crockford => &crate::dec::CROCKFORD_LUT,
        Geohash => &crate::dec::GEOHASH_LUT,
        Z => &crate::dec::Z_LUT,
        _ => core::hint::unreachable_unchecked(),
    };

    let lut_0_63 = _mm512_loadu_si512(lut.as_ptr() as *const _);
    let lut_64_127 = _mm512_loadu_si512(lut.as_ptr().offset(64) as *const _);
    let mask_ge_64 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(64u8 as i8));

    let v_0_63 = _mm512_permutexvar_epi8(src, lut_0_63);
    let v_64_127 = _mm512_permutexvar_epi8(src, lut_64_127);

    _mm512_mask_blend_epi8(mask_ge_64, v_0_63, v_64_127)
}

#[inline(always)]
pub unsafe fn padcount_avx512(src: &[u8]) -> usize {
    debug_assert_eq!(src.len(), 8);
    _popcnt32(_mm_cmpeq_epi8_mask(
        _mm_loadl_epi64(src.as_ptr() as *const _),
        _mm_set1_epi8(b'=' as i8),
    ) as i32) as usize
}

pub(crate) unsafe fn b32dec_avx512<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) {
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 64 {
        let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const __m512i);
        let d = from_char_avx512::<A>(s);

        let shifts8 = _mm512_set_epi8(
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
            1, 32, 1, 32, 1, 32, 1, 32,
        );

        let s16 = _mm512_maddubs_epi16(d, shifts8);

        let shifts16 = _mm512_set_epi16(
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
            1, 1024, 1, 1024,
        );

        let s32 = _mm512_madd_epi16(s16, shifts16);
        let s32_shuffled = _mm512_shuffle_epi32(s32, 0b10_11_00_01);
        let s32_shifted = _mm512_srl_epi64(s32_shuffled, _mm_set1_epi64x(12));

        let mask = _mm512_set1_epi64(0b0000000000001111111111111111111111111111111100000000000000000000);
        let blitted = _mm512_ternarylogic_epi64(mask, s32_shifted, s32_shuffled, 0xca);

        let perm = _mm512_set_epi8(
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            56, 57, 58, 59, 60,
            48, 49, 50, 51, 52,
            40, 41, 42, 43, 44,
            32, 33, 34, 35, 36,
            24, 25, 26, 27, 28,
            16, 17, 18, 19, 20,
            8, 9, 10, 11, 12,
            0, 1, 2, 3, 4,
        );

        let res = _mm512_maskz_permutexvar_epi8(0x000000FFFFFFFFFF, __m512i::from(perm), blitted);
        _mm512_mask_storeu_epi8(dst.as_ptr().add(dst_cur) as *mut i8, 0x000000FFFFFFFFFF, res);

        src_cur += 64;
        dst_cur += 40;
    }
}

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;
    use crate::{RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS};

    static FROM_CHAR_INPUT: [u8; 64] = [
        b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9',
        b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z',
        b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
        0, u8::MAX,
    ];

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

    fn generate_expected_from_char_output(input: &[u8; 64], alphabet: &[u8; 32]) -> [u8; 64] {
        let mut ret = [0u8; 64];
        for i in 0..64 {
            ret[i] = expected_from_char(input[i], alphabet);
        }
        ret
    }

    #[test]
    fn test_from_char_avx512_rfc4648() {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            let result_reg = from_char_avx512::<Rfc4648>(src_reg);
            let mut actual_output_bytes = [0u8; 64];
            _mm512_storeu_si512(actual_output_bytes.as_mut_ptr() as *mut _, result_reg);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, RFC4648_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "AVX-512 Rfc4648 from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_avx512_rfc4648hex() {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            let result_reg = from_char_avx512::<Rfc4648Hex>(src_reg);
            let mut actual_output_bytes = [0u8; 64];
            _mm512_storeu_si512(actual_output_bytes.as_mut_ptr() as *mut _, result_reg);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, RFC4648HEX_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "AVX-512 Rfc4648Hex from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_avx512_crockford() {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            let result_reg = from_char_avx512::<Crockford>(src_reg);
            let mut actual_output_bytes = [0u8; 64];
            _mm512_storeu_si512(actual_output_bytes.as_mut_ptr() as *mut _, result_reg);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, CROCKFORD_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "AVX-512 Crockford from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_avx512_geohash() {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            let result_reg = from_char_avx512::<Geohash>(src_reg);
            let mut actual_output_bytes = [0u8; 64];
            _mm512_storeu_si512(actual_output_bytes.as_mut_ptr() as *mut _, result_reg);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, GEOHASH_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "AVX-512 Geohash from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_avx512_z() {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            let result_reg = from_char_avx512::<Z>(src_reg);
            let mut actual_output_bytes = [0u8; 64];
            _mm512_storeu_si512(actual_output_bytes.as_mut_ptr() as *mut _, result_reg);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, Z_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "AVX-512 Z from_char mismatch");
        }
    }

    #[test]
    fn test_padcount_avx512_none() {
        let src: [u8; 8] = *b"ABCDEFGH";
        let count = unsafe { padcount_avx512(&src) };
        assert_eq!(count, 0);
    }

    #[test]
    fn test_padcount_avx512_one() {
        let src: [u8; 8] = *b"ABCDEFG=";
        let count = unsafe { padcount_avx512(&src) };
        assert_eq!(count, 1);
    }

    #[test]
    fn test_padcount_avx512_two() {
        let src: [u8; 8] = *b"ABCDEF==";
        let count = unsafe { padcount_avx512(&src) };
        assert_eq!(count, 2);
    }

    #[test]
    fn test_padcount_avx512_all() {
        let src: [u8; 8] = *b"========";
        let count = unsafe { padcount_avx512(&src) };
        assert_eq!(count, 8);
    }

    #[bench]
    fn bench_from_char_avx512(b: &mut Bencher) {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            b.iter(|| {
                black_box(from_char_avx512::<Z>(black_box(src_reg)));
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
    fn bench_padcount_avx512(b: &mut Bencher) {
        b.iter(|| {
            for input in PADCOUNT_INPUT.iter() {
                unsafe { black_box(padcount_avx512(black_box(input))) };
            }
        });
    }

    #[bench]
    fn bench_b32dec_avx512(b: &mut Bencher) {
        let input = b"GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";
        let mut output = [0u8; 40];
        b.iter(|| {
            unsafe { black_box(b32dec_avx512::<Z>(black_box(input), black_box(&mut output))) };
        });
    }

    #[bench]
    fn bench_b32dec_avx512_bulk(b: &mut Bencher) {
        let mut input = vec![0u8; 16777216];
        let mut output = vec![0u8; 10485760];
        for (i, b) in input.iter_mut().enumerate() {
            *b = b"GEZDGNBVGY3TQOJQ"[i % 16];
        }
        b.iter(|| {
            unsafe { black_box(b32dec_avx512::<Z>(black_box(&input), black_box(&mut output))) };
        });
    }
}
