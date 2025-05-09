use std::mem::transmute;
#[cfg(feature = "simd")]
use std::simd::{Simd};
#[cfg(feature = "simd")]
use std::simd::cmp::{SimdPartialOrd};
#[cfg(feature = "avx-512")]
use std::arch::x86_64::*;

use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z, RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS};

const fn generate_decode_lut(alphabet: &[u8; 32]) -> [u8; 256] {
    let mut lut = [u8::MAX; 256];
    let mut i = 0u8;
    while i < 32 {
        let char_code = alphabet[i as usize];
        lut[char_code.to_ascii_lowercase() as usize] = i;
        lut[char_code.to_ascii_uppercase() as usize] = i;
        i += 1;
    }
    lut[b'=' as usize] = 0;
    lut
}

const RFC4648_LUT: [u8; 256] = generate_decode_lut(RFC4648_CHARS);
const RFC4648HEX_LUT: [u8; 256] = generate_decode_lut(RFC4648HEX_CHARS);
const CROCKFORD_LUT: [u8; 256] = generate_decode_lut(CROCKFORD_CHARS);
const GEOHASH_LUT: [u8; 256] = generate_decode_lut(GEOHASH_CHARS);
const Z_LUT: [u8; 256] = generate_decode_lut(Z_CHARS);

#[inline(always)]
unsafe fn from_char<const A: u8>(value: u8) -> u8 {
    match A {
        Rfc4648 => RFC4648_LUT[value as usize],
        Rfc4648Hex => RFC4648HEX_LUT[value as usize],
        Crockford => CROCKFORD_LUT[value as usize],
        Geohash => GEOHASH_LUT[value as usize],
        Z => Z_LUT[value as usize],
        _ => core::hint::unreachable_unchecked(),
    }
}

#[inline(always)]
#[cfg(feature = "avx-512")]
unsafe fn from_char_avx512<const A: u8>(src: __m512i) -> __m512i {
    let lut = match A {
        Rfc4648 => RFC4648_LUT,
        Rfc4648Hex => RFC4648HEX_LUT,
        Crockford => CROCKFORD_LUT,
        Geohash => GEOHASH_LUT,
        Z => Z_LUT,
        _ => core::hint::unreachable_unchecked(),
    };

    let lut_0_63 = _mm512_loadu_si512(lut.as_ptr() as *const _);
    let lut_64_127 = _mm512_loadu_si512(lut.as_ptr().offset(64) as *const _);
    let mask_ge_64 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(64u8 as i8));

    // vpermb only takes the first 5 bits of each lane in the index vector,
    // so no need to subtract 64 from src for lut_64_127
    let v_0_63 = _mm512_permutexvar_epi8(src, lut_0_63);
    let v_64_127 = _mm512_permutexvar_epi8(src, lut_64_127);

    _mm512_mask_blend_epi8(mask_ge_64, v_0_63, v_64_127)
}

#[cfg(feature = "simd")]
unsafe fn from_char_simd<const A: u8>(src: Simd<u8, 64>) -> Simd<u8, 64> {
    let lut = match A {
        Rfc4648 => RFC4648_LUT,
        Rfc4648Hex => RFC4648HEX_LUT,
        Crockford => CROCKFORD_LUT,
        Geohash => GEOHASH_LUT,
        Z => Z_LUT,
        _ => core::hint::unreachable_unchecked(),
    };

    let lut_0_63 = transmute::<_, *const Simd<u8, 64>>(lut.as_ptr().add(0)).read_unaligned();
    let lut_64_127 = transmute::<_, *const Simd<u8, 64>>(lut.as_ptr().add(64)).read_unaligned();
    let mask_ge_64 = src.simd_ge(Simd::splat(64));

    let v_0_63 = lut_0_63.swizzle_dyn(src);
    let v_64_127 = lut_64_127.swizzle_dyn(src & Simd::splat(0x3F)); // Hack to make 255 work - probably never encountered

    mask_ge_64.select(v_64_127, v_0_63)
}

#[inline(always)]
#[cfg(feature = "avx-512")]
pub unsafe fn padcount_avx512(src: &[u8]) -> usize {
    debug_assert_eq!(src.len(), 8);
    _popcnt32(_mm_cmpeq_epi8_mask(
        _mm_loadl_epi64(src.as_ptr() as *const _),
        _mm_set1_epi8(b'=' as i8),
    ) as i32) as usize
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

#[cfg(feature = "avx-512")]
unsafe fn b32dec_avx512<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) {
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 64 {
        let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const i32);
        let d = from_char_avx512::<A>(s);
        // -> 000eeeee 000ddeee 000ddddd 000ccccd 000bcccc 000bbbbb 000aaabb 000aaaaa

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

        let s16 = _mm512_maddubs_epi16(d, shifts8); // b1 | b2 << 5
        // -> 000000ddeeeeeeee 000000ccccdddddd 000000bbbbbbcccc 000000aaaaaaaabb

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

        let s32 = _mm512_madd_epi16(s16, shifts16); // w1 | w2 << 10
        // -> 000000000000ccccddddddddeeeeeeee 000000000000aaaaaaaabbbbbbbbcccc
        let s32_shuffled = _mm512_shuffle_epi32(s32, 0b10_11_00_01);
        // -> 000000000000aaaaaaaabbbbbbbbcccc 000000000000ccccddddddddeeeeeeee
        let s32_shifted = _mm512_srl_epi64(s32_shuffled, _mm_set1_epi64x(12));
        // -> 000000000000000000000000aaaaaaaa bbbbbbbbcccc000000000000ccccdddd

        let mask = _mm512_set1_epi64(0b0000000000001111111111111111111111111111111100000000000000000000);
        // select bits from s32s if mask = 1, else s32ss
        let blitted = _mm512_ternarylogic_epi64(mask, s32_shifted, s32_shuffled, 0xca);

        let perm = _mm512_set_epi8( // byteswap 40 bytes, then 20 zeroes
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

pub fn b32dec<'a>(src: &'a [u8], dst: &'a mut [u8], alphabet: u8) -> &'a [u8] {
    if src.len() == 0 {
        return &dst[0..0];
    }

    if dst.len() < ((src.len() + 3) / 8) * 5 {
        panic!("destination buffer too small");
    }

    unsafe {
        match alphabet {
            Rfc4648 => b32dec_generic::<Rfc4648>(src, dst),
            Rfc4648Hex => b32dec_generic::<Rfc4648Hex>(src, dst),
            Crockford => b32dec_generic::<Crockford>(src, dst),
            Geohash => b32dec_generic::<Geohash>(src, dst),
            Z => b32dec_generic::<Z>(src, dst),
            _ => panic!("invalid alphabet selected"),
        }
    }
}

pub unsafe fn b32dec_generic<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    if src.len() >= 64 {
        #[cfg(feature = "avx-512")] {
            b32dec_avx512::<A>(src, dst);
        }
        // #[cfg(feature = "simd")] {
        //     b32dec_simd::<A>(src, dst);
        // }
    }

    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let src_tail = src.len() - src.len() % 64;
    #[cfg(any(feature = "avx-512", feature = "simd"))]
    let dst_tail = dst.len() - dst.len() % 40;

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

        let data0 = from_char::<A>(padded_chunk[0]);
        let data1 = from_char::<A>(padded_chunk[1]);
        let data2 = from_char::<A>(padded_chunk[2]);
        let data3 = from_char::<A>(padded_chunk[3]);
        let data4 = from_char::<A>(padded_chunk[4]);
        let data5 = from_char::<A>(padded_chunk[5]);
        let data6 = from_char::<A>(padded_chunk[6]);
        let data7 = from_char::<A>(padded_chunk[7]);

        dst_chunk[0] = (data0 << 3) | (data1 >> 2);
        dst_chunk[1] = (data1 << 6) | (data2 << 1) | (data3 >> 4);
        dst_chunk[2] = (data3 << 4) | (data4 >> 1);
        dst_chunk[3] = (data4 << 7) | (data5 << 2) | (data6 >> 3);
        dst_chunk[4] = (data6 << 5) | data7;
    }

    return &dst[..dst.len() - 5 + (8 - pad_count) * 5 / 8]
}

#[cfg(test)] extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;

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
                let actual = from_char::<Rfc4648>(value);
                assert_eq!(actual, expected, "Rfc4648 from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_rfc4648hex() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, RFC4648HEX_CHARS);
            unsafe {
                let actual = from_char::<Rfc4648Hex>(value);
                assert_eq!(actual, expected, "Rfc4648Hex from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_crockford() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, CROCKFORD_CHARS);
            unsafe {
                let actual = from_char::<Crockford>(value);
                assert_eq!(actual, expected, "Crockford from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_geohash() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, GEOHASH_CHARS);
            unsafe {
                let actual = from_char::<Geohash>(value);
                assert_eq!(actual, expected, "Geohash from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    #[test]
    fn test_from_char_scalar_z() {
        for value in 0..=255u8 {
            let expected = expected_from_char(value, Z_CHARS);
            unsafe {
                let actual = from_char::<Z>(value);
                assert_eq!(actual, expected, "Z from_char mismatch for value {} ({})", value, value as char);
            }
        }
    }

    static FROM_CHAR_INPUT: [u8; 64] = [
        b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9',
        b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z',
        b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
        0, u8::MAX,
    ];

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
    fn test_from_char_simd_rfc4648() {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            let result_simd = from_char_simd::<Rfc4648>(src_simd);
            let mut actual_output_bytes = [0u8; 64];
            result_simd.copy_to_slice(&mut actual_output_bytes);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, RFC4648_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "SIMD Rfc4648 from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_simd_rfc4648hex() {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            let result_simd = from_char_simd::<Rfc4648Hex>(src_simd);
            let mut actual_output_bytes = [0u8; 64];
            result_simd.copy_to_slice(&mut actual_output_bytes);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, RFC4648HEX_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "SIMD Rfc4648Hex from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_simd_crockford() {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            let result_simd = from_char_simd::<Crockford>(src_simd);
            let mut actual_output_bytes = [0u8; 64];
            result_simd.copy_to_slice(&mut actual_output_bytes);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, CROCKFORD_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "SIMD Crockford from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_simd_geohash() {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            let result_simd = from_char_simd::<Geohash>(src_simd);
            let mut actual_output_bytes = [0u8; 64];
            result_simd.copy_to_slice(&mut actual_output_bytes);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, GEOHASH_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "SIMD Geohash from_char mismatch");
        }
    }

    #[test]
    fn test_from_char_simd_z() {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            let result_simd = from_char_simd::<Z>(src_simd);
            let mut actual_output_bytes = [0u8; 64];
            result_simd.copy_to_slice(&mut actual_output_bytes);

            let expected_output_bytes = generate_expected_from_char_output(&FROM_CHAR_INPUT, Z_CHARS);
            assert_eq!(&actual_output_bytes[..], &expected_output_bytes[..], "SIMD Z from_char mismatch");
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
    fn bench_from_char_avx512(b: &mut Bencher) {
        unsafe {
            let src_reg = _mm512_loadu_si512(FROM_CHAR_INPUT.as_ptr() as *const _);
            b.iter(|| {
                black_box(from_char_avx512::<Z>(black_box(src_reg)));
            });
        }
    }

    #[bench]
    fn bench_from_char_simd(b: &mut Bencher) {
        unsafe {
            let src_simd = Simd::<u8, 64>::from_slice(&FROM_CHAR_INPUT);
            b.iter(|| {
                black_box(from_char_simd::<Z>(black_box(src_simd)));
            });
        }
    }

    #[bench]
    fn bench_from_char(b: &mut Bencher) {
        unsafe {
            b.iter(|| {
                for src in FROM_CHAR_INPUT.iter() {
                    black_box(from_char::<Z>(black_box(*src)));
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
            black_box(b32dec(black_box(input), black_box(&mut output), Z));
        });
    }

    #[bench]
    fn bench_b32dec(b: &mut Bencher) {
        let input = b"GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBV";
        let mut output = [0u8; 35];
        b.iter(|| {
            black_box(b32dec(black_box(input), black_box(&mut output), Z));
        });
    }

    use base32::{Alphabet, encode, decode};

    #[test]
    fn test_b32dec_rfc4648_padding_empty() {
        let input = "".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, "").unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_padding_full_block() {
        let input = "ORSXG5DJORUXG5LNORUWYZLSEBFWC2LTN5ZG64DDMNWGC2LPN5ZG64TON5XHIZLE".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, "ORSXG5DJORUXG5LNORUWYZLSEBFWC2LTN5ZG64DDMNWGC2LPN5ZG64TON5XHIZLE").unwrap();
        let mut dst = vec![0u8; expected_output.len()];
        b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648hex_various_lengths_padding() {
        let alphabet_crate = Alphabet::Rfc4648Hex { padding: true };
        let alphabet_u8 = Rfc4648Hex;

        for data_len in 0..=255 {
            let data: Vec<u8> = (0..data_len).map(|i| i as u8).collect();
            let encoded = encode(alphabet_crate, &data);
            let expected = decode(alphabet_crate, &encoded).unwrap();
            let mut dst = vec![0u8; (expected.len() + 4) / 5 * 5];
            let dst = b32dec(encoded.as_bytes(), &mut dst, alphabet_u8);
            assert_eq!(dst, expected, "failed for length: {}", data_len);
        }
    }

    #[test]
    fn test_b32dec_rfc4648_boundary() {
        let alphabet_crate = Alphabet::Rfc4648 { padding: false };
        let alphabet_u8 = Rfc4648;

        for data_len in 60..70 { // Test around the 64-byte boundary
            let data: Vec<u8> = (0..data_len).map(|i| i as u8).collect();
            let encoded = encode(alphabet_crate, &data);
            let expected = decode(alphabet_crate, &encoded).unwrap();
            let mut dst = vec![0u8; (expected.len() + 4) / 5 * 5];
            let dst = b32dec(encoded.as_bytes(), &mut dst, alphabet_u8);
            assert_eq!(dst, expected, "failed for length {}", data_len);
        }
    }

    #[test]
    fn test_b32dec_rfc4648_boundary_padding() {
        let alphabet_crate = Alphabet::Rfc4648 { padding: true };
        let alphabet_u8 = Rfc4648;

        for data_len in 60..70 { // Test around the 64-byte boundary
            let data: Vec<u8> = (0..data_len).map(|i| i as u8).collect();
            let encoded = encode(alphabet_crate, &data);
            let expected = decode(alphabet_crate, &encoded).unwrap();
            let mut dst = vec![0u8; (expected.len() + 4) / 5 * 5];
            let dst = b32dec(encoded.as_bytes(), &mut dst, alphabet_u8);
            assert_eq!(dst, expected, "failed for length {}", data_len);
        }
    }

    #[test]
    #[should_panic(expected = "destination buffer too small")]
    fn test_b32dec_destination_buffer_too_small() {
        let data = b"foobar";
        let encoded = encode(Alphabet::Rfc4648 { padding: true }, data);
        let mut dst = vec![0u8; 1];
        b32dec(encoded.as_bytes(), &mut dst, Rfc4648);
    }

    #[test]
    #[should_panic(expected = "invalid alphabet selected")]
    fn test_b32dec_invalid_alphabet() {
        let data = b"foobar";
        let encoded = encode(Alphabet::Rfc4648 { padding: true }, data);
        let mut dst = vec![0u8; 10];
        b32dec(encoded.as_bytes(), &mut dst, 99);
    }

    #[test]
    fn test_b32dec_rfc4648_tail_padding_6() {
        let input = "AA======";
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, input).unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input.as_bytes(), &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_tail_padding_5() {
        let input = "ABQ=====";
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, input).unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input.as_bytes(), &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_tail_padding_4() {
        let input = "ABQY====";
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, input).unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input.as_bytes(), &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_tail_padding_2() {
        let input = "ABQYIC==";
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, input).unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input.as_bytes(), &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_tail_nopadding() {
        let input = "ABQYICAA";
        let expected_output = decode(Alphabet::Rfc4648 { padding: true }, input).unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input.as_bytes(), &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_1_byte() {
        let input = "AE".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: false }, "AE").unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_2_bytes() {
        let input = "AEBA".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: false }, "AEBA").unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_3_bytes() {
        let input = "AEBAG".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: false }, "AEBAG").unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_rfc4648_4_bytes() {
        let input = "AEBAGCA".as_bytes();
        let expected_output = decode(Alphabet::Rfc4648 { padding: false }, "AEBAGCA").unwrap();
        let mut dst = vec![0u8; 5];
        let dst = b32dec(input, &mut dst, Rfc4648);
        assert_eq!(dst, expected_output);
    }

    #[test]
    fn test_b32dec_crockford_mixed_case() {
        let alphabet_u8 = Crockford;
        let encoded_base = "CRRSGZDB";
        let encoded_lower = encoded_base.to_lowercase();
        let encoded_upper = encoded_base.to_uppercase();
        let encoded_mixed: String = encoded_base.chars().enumerate().map(|(i, c)| {
            if i % 2 == 0 { c.to_uppercase().next() } else { c.to_lowercase().next() }.unwrap()
        }).collect();


        let expected = decode(Alphabet::Crockford, encoded_base).unwrap(); // Base32 crate handles case-insensitivity for Crockford

        let mut dst_lower = vec![0u8; expected.len()];
        b32dec(encoded_lower.as_bytes(), &mut dst_lower, alphabet_u8);
        assert_eq!(dst_lower, expected, "failed lowercase decoding");

        let mut dst_upper = vec![0u8; expected.len()];
        b32dec(encoded_upper.as_bytes(), &mut dst_upper, alphabet_u8);
        assert_eq!(dst_upper, expected, "failed uppercase decoding");

        let mut dst_mixed = vec![0u8; expected.len()];
        b32dec(encoded_mixed.as_bytes(), &mut dst_mixed, alphabet_u8);
        assert_eq!(dst_mixed, expected, "failed mixed case decoding");
    }

}
