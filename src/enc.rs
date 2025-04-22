use std::simd::{Simd, Mask};
use std::simd::cmp::SimdPartialOrd;
use std::mem::transmute;
use std::arch::x86_64::*;

use super::{
    Rfc4648, Rfc4648Hex, Crockford, Geohash, Z,
    RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS
};

static RFC4648_LUT: [u8; 64] = [
    b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P',
    b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z', b'2', b'3', b'4', b'5', b'6', b'7',
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

static RFC4648HEX_LUT: [u8; 64] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P', b'Q', b'R', b'S', b'T', b'U', b'V',
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

static CROCKFORD_LUT: [u8; 64] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'A', b'B', b'C', b'D', b'E', b'F',
    b'G', b'H', b'J', b'K', b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'X', b'Y', b'Z',
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

static GEOHASH_LUT: [u8; 64] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'b', b'c', b'd', b'e', b'f', b'g',
    b'h', b'j', b'k', b'm', b'n', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

static Z_LUT: [u8; 64] = [
    b'y', b'b', b'n', b'd', b'r', b'f', b'g', b'8', b'e', b'j', b'k', b'm', b'c', b'p', b'q', b'x',
    b'o', b't', b'1', b'u', b'w', b'i', b's', b'z', b'a', b'3', b'4', b'5', b'h', b'7', b'6', b'9',
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[inline(always)]
unsafe fn to_char<const A: u8>(value: u8) -> u8 {
    match A {
        Rfc4648 => RFC4648_CHARS[value as usize],
        Rfc4648Hex => RFC4648HEX_CHARS[value as usize],
        Crockford => CROCKFORD_CHARS[value as usize],
        Geohash => GEOHASH_CHARS[value as usize],
        Z => Z_CHARS[value as usize],
        _ => core::hint::unreachable_unchecked(), 
    }
}

unsafe fn to_char_avx512<const A: u8>(src: __m512i) -> __m512i {
    let lut = match A {
        Rfc4648 => RFC4648_LUT,
        Rfc4648Hex => RFC4648HEX_LUT,
        Crockford => CROCKFORD_LUT,
        Geohash => GEOHASH_LUT,
        Z => Z_LUT,
        _ => core::hint::unreachable_unchecked(), 
    };
    
    let lut_reg = _mm512_loadu_si512(lut.as_ptr() as *const _);
    _mm512_permutexvar_epi8(src, lut_reg)
}

#[inline(always)]
unsafe fn to_char_simd<const A: u8>(src: Simd<u8, 64>) -> Simd<u8, 64> {
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
        _ => core::hint::unreachable_unchecked(),        
    }
}

pub fn b32enc(src: &[u8], dst: &mut [u8], alphabet: u8) {
    if dst.len() < ((src.len() + 4) / 5) * 8 {
        panic!("destination buffer too small");
    }

    unsafe { 
        match alphabet {
            Rfc4648 => b32enc_generic::<Rfc4648>(src, dst),
            Rfc4648Hex => b32enc_generic::<Rfc4648Hex>(src, dst),
            Crockford => b32enc_generic::<Crockford>(src, dst),
            Geohash => b32enc_generic::<Geohash>(src, dst),
            Z => b32enc_generic::<Z>(src, dst),
            _ => panic!("invalid alphabet selected"),
        }
    }
}

unsafe fn b32enc_generic<const A: u8>(src: &[u8], dst: &mut [u8]) {
    let simd_src_len = (src.len() / 40) * 40;
    let simd_dst_len = (simd_src_len / 40) * 64;
    if simd_src_len > 0 {
        b32enc_avx512::<A>(&src[..simd_src_len], &mut dst[..simd_dst_len]);
    }

    let rem_src = &src[simd_src_len..];
    let rem_dst = &mut dst[simd_dst_len..];
    let mut rem_dst_cur = 0;
    for src_chunk in rem_src.chunks(5) {
        let dst_chunk = &mut rem_dst[rem_dst_cur..];
        let mut padded_chunk = [0u8; 5];
        padded_chunk[..src_chunk.len()].copy_from_slice(src_chunk);

        dst_chunk[0] = to_char::<A>((padded_chunk[0] & 0xf8) >> 3);
        dst_chunk[1] = to_char::<A>(((padded_chunk[0] & 0x07) << 2) | ((padded_chunk[1] & 0xC0) >> 6));
        dst_chunk[2] = to_char::<A>((padded_chunk[1] & 0x3E) >> 1);
        dst_chunk[3] = to_char::<A>(((padded_chunk[1] & 0x01) << 4) | ((padded_chunk[2] & 0xF0) >> 4));
        dst_chunk[4] = to_char::<A>(((padded_chunk[2] & 0x0F) << 1) | (padded_chunk[3] >> 7));
        dst_chunk[5] = to_char::<A>((padded_chunk[3] & 0x7C) >> 2);
        dst_chunk[6] = to_char::<A>(((padded_chunk[3] & 0x03) << 3) | ((padded_chunk[4] & 0xE0) >> 5));
        dst_chunk[7] = to_char::<A>(padded_chunk[4] & 0x1F);

        let dst_len = (src_chunk.len() * 8 + 4) / 5; // ceil(src_chunk.len() * 8 / 5)
        for i in dst_len..8 {
            dst_chunk[i] = b'=';
        }

        rem_dst_cur += 8;
    }
}

fn b32enc_avx512<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 40 {
        unsafe {
            let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const i32);
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
        }
        src_cur += 40;
        dst_cur += 64;
    }
    return dst;
}

#[inline(always)]
unsafe fn b32enc_simd<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
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
        let s = transmute::<_, *const Simd<u8, 64>>(src.as_ptr().add(src_cur)).read_unaligned();
        let p = transmute::<_, Simd<u64, 8>>(s.swizzle_dyn(shuf));
        let d = (p >> Simd::splat(3)) & Simd::splat(0x1F00000000000000)
            | (p >> Simd::splat(6)) & Simd::splat(0x001F000000000000)
            | (p >> Simd::splat(9)) & Simd::splat(0x00001F0000000000)
            | (p >> Simd::splat(12)) & Simd::splat(0x0000001F00000000)
            | (p >> Simd::splat(15)) & Simd::splat(0x000000001F000000)
            | (p >> Simd::splat(18)) & Simd::splat(0x00000000001F0000)
            | (p >> Simd::splat(21)) & Simd::splat(0x0000000000001F00)
            | (p >> Simd::splat(24)) & Simd::splat(0x000000000000001F);

        let db = transmute::<_, Simd<u8, 64>>(d).swizzle_dyn(endian64);
        let res: Simd<u8, 64> = to_char_simd::<A>(db);

        transmute::<_, *mut Simd<u8, 64>>(dst.as_ptr().add(dst_cur)).write_unaligned(res);
        src_cur += 40;
        dst_cur += 64;
    }

    return dst;
}

#[cfg(test)] extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use test::bench::Bencher;
    use std::hint::black_box;
    use base32::encode;
    use base32::Alphabet;

    fn encoded_len(input_len: usize) -> usize {
        ((input_len + 4) / 5) * 8
    }

    #[test]
    fn test_b32enc_empty() {
        let src = b"";
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }

    #[test]
    fn test_b32enc_less_than_5_bytes() {

        let src1 = b"f";
        let expected1_len = encoded_len(src1.len());
        let mut dst1 = vec![0u8; expected1_len];
        b32enc(src1, &mut dst1, Rfc4648);
        let expected1 = encode(Alphabet::Rfc4648 { padding: true }, src1);
        assert_eq!(std::str::from_utf8(&dst1).unwrap(), expected1);

        let src2 = b"fo";
        let expected2_len = encoded_len(src2.len());
        let mut dst2 = vec![0u8; expected2_len];
        b32enc(src2, &mut dst2, Rfc4648);
        let expected2 = encode(Alphabet::Rfc4648 { padding: true }, src2);
        assert_eq!(std::str::from_utf8(&dst2).unwrap(), expected2);

        let src3 = b"foo";
        let expected3_len = encoded_len(src3.len());
        let mut dst3 = vec![0u8; expected3_len];
        b32enc(src3, &mut dst3, Rfc4648);
        let expected3 = encode(Alphabet::Rfc4648 { padding: true }, src3);
        assert_eq!(std::str::from_utf8(&dst3).unwrap(), expected3);

        let src4 = b"foob";
        let expected4_len = encoded_len(src4.len());
        let mut dst4 = vec![0u8; expected4_len];
        b32enc(src4, &mut dst4, Rfc4648);
        let expected4 = encode(Alphabet::Rfc4648 { padding: true }, src4);
        assert_eq!(std::str::from_utf8(&dst4).unwrap(), expected4);
    }

    #[test]
    fn test_b32enc_exact_5_bytes() {
        let src = b"fooba";
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }

    #[test]
    fn test_b32enc_multiple_of_5_bytes() {
        let src = b"foobarfoobar";
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }

    #[test]
    fn test_b32enc_input_around_40_bytes() {

        let src1 = b"01234567890123456789012345678901234567";
        let expected1_len = encoded_len(src1.len());
        let mut dst1 = vec![0u8; expected1_len];
        b32enc(src1, &mut dst1, Rfc4648);
        let expected1 = encode(Alphabet::Rfc4648 { padding: true }, src1);
        assert_eq!(std::str::from_utf8(&dst1).unwrap(), expected1);

        let src2 = b"0123456789012345678901234567890123456789";
        let expected2_len = encoded_len(src2.len());
        let mut dst2 = vec![0u8; expected2_len];
        b32enc(src2, &mut dst2, Rfc4648);
        let expected2 = encode(Alphabet::Rfc4648 { padding: true }, src2);
        assert_eq!(std::str::from_utf8(&dst2).unwrap(), expected2);

        let src3 = b"0123456789012345678901234567890123456789abcde";
        let expected3_len = encoded_len(src3.len());
        let mut dst3 = vec![0u8; expected3_len];
        b32enc(src3, &mut dst3, Rfc4648);
        let expected3 = encode(Alphabet::Rfc4648 { padding: true }, src3);
        assert_eq!(std::str::from_utf8(&dst3).unwrap(), expected3);
    }

    #[test]
    fn test_b32enc_long_input() {
        let src = b"This is a longer test string to ensure that both AVX-512 and the tail handling work correctly for inputs significantly larger than 40 bytes. This is a longer test string to ensure that both AVX-512 and the tail handling work correctly for inputs significantly larger than 40 bytes. This is a longer test string to ensure that both AVX-512 and the tail handling work correctly for inputs significantly larger than 40 bytes. This is a longer test string to ensure that both AVX-512 and the tail handling work correctly for inputs significantly larger than 40 bytes. This is a longer test string to ensure that both AVX-512 and the tail handling work correctly for inputs significantly larger than 40 bytes.";
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }

    #[test]
    fn test_b32enc_input_with_various_tail_lengths() {
        for i in 1..40 {
            let mut src_base = vec![b'A'; 40];
            let tail_bytes = vec![b'B'; i];
            src_base.extend_from_slice(&tail_bytes);
            let src = src_base;
            let expected_len = encoded_len(src.len());
            let mut dst = vec![0u8; expected_len];
            b32enc(&src, &mut dst, Rfc4648);
            let expected = encode(Alphabet::Rfc4648 { padding: true }, &src);
            assert_eq!(std::str::from_utf8(&dst).unwrap(), expected, "Failed for input length {}", src.len());
        }

        for i in 1..5 {
            let mut src_base = vec![b'C'; 40];
            let tail_bytes = vec![b'D'; i];
            src_base.extend_from_slice(&tail_bytes);
            let src = src_base;
            let expected_len = encoded_len(src.len());
            let mut dst = vec![0u8; expected_len];
            b32enc(&src, &mut dst, Rfc4648);
            let expected = encode(Alphabet::Rfc4648 { padding: true }, &src);
            assert_eq!(std::str::from_utf8(&dst).unwrap(), expected, "Failed for input length {}", src.len());
        }
    }

    #[test]
    fn test_b32enc_all_zeroes() {
        let src = vec![0u8; 50];
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(&src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, &src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }

    #[test]
    fn test_b32enc_all_ones() {
        let src = vec![0xFFu8; 50];
        let expected_len = encoded_len(src.len());
        let mut dst = vec![0u8; expected_len];
        b32enc(&src, &mut dst, Rfc4648);
        let expected = encode(Alphabet::Rfc4648 { padding: true }, &src);
        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected);
    }
    
    // Define the static input data with values 0-31 in the first half and zeros in the second half
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


    #[test]
    fn test_rfc4648_simd() {
        unsafe {
            let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
            let result_simd = to_char_simd::<Rfc4648>(src);
            let mut actual = [0u8; 64];
            result_simd.copy_to_slice(&mut actual);
            assert_eq!(&actual[0..32], RFC4648_CHARS);
        }
    }

    #[test]
    fn test_rfc4648hex_simd() {
        unsafe {
            let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
            let result_simd = to_char_simd::<Rfc4648Hex>(src);
            let mut actual = [0u8; 64];
            result_simd.copy_to_slice(&mut actual);
            assert_eq!(&actual[0..32], RFC4648HEX_CHARS);
        }
    }

    #[test]
    fn test_crockford_simd() {
        unsafe {
            let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
            let result_simd = to_char_simd::<Crockford>(src);
            let mut actual = [0u8; 64];
            result_simd.copy_to_slice(&mut actual);
            assert_eq!(&actual[0..32], CROCKFORD_CHARS);
        }
    }

    #[test]
    fn test_geohash_simd() {
        unsafe {
            let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
            let result_simd = to_char_simd::<Geohash>(src);
            let mut actual = [0u8; 64];
            result_simd.copy_to_slice(&mut actual);
            assert_eq!(&actual[0..32], GEOHASH_CHARS);
        }
    }

    #[test]
    fn test_z_simd() {
        unsafe {
            let src = Simd::<u8, 64>::from_slice(&TO_CHAR_INPUT);
            let result = to_char_simd::<Z>(src);
            let mut actual = [0u8; 64];
            result.copy_to_slice(&mut actual);
            assert_eq!(&actual[0..32], Z_CHARS);
        }
    }

    #[test]
    fn test_to_char_scalar_rfc4648() {
        unsafe { 
            for value in 0..32u8 {
                let expected = RFC4648_CHARS[value as usize];
                let actual = to_char::<Rfc4648>(value);
                assert_eq!(actual, expected, "Rfc4648 mismatch for value {}", value);
            }
        }
    }

    #[test]
    fn test_to_char_scalar_rfc4648hex() {
        unsafe { 
            for value in 0..32u8 {
                let expected = RFC4648HEX_CHARS[value as usize];
                let actual = to_char::<Rfc4648Hex>(value);
                assert_eq!(actual, expected, "Rfc4648Hex mismatch for value {}", value);
            }
        }
    }

    #[test]
    fn test_to_char_scalar_crockford() {
        unsafe { 
            for value in 0..32u8 {
                let expected = CROCKFORD_CHARS[value as usize];
                let actual = to_char::<Crockford>(value);
                assert_eq!(actual, expected, "Crockford mismatch for value {}", value);
            }
        }
    }

    #[test]
    fn test_to_char_scalar_geohash() {
        unsafe { 
            for value in 0..32u8 {
                let expected = GEOHASH_CHARS[value as usize];
                let actual = to_char::<Geohash>(value);
                assert_eq!(actual, expected, "Geohash mismatch for value {}", value);
            }
        }
    }

    #[test]
    fn test_to_char_scalar_z() {
        unsafe { 
            for value in 0..32u8 {
                let expected = Z_CHARS[value as usize];
                let actual = to_char::<Z>(value);
                assert_eq!(actual, expected, "Z mismatch for value {}", value);
            }
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
            unsafe { black_box(b32enc_avx512::<Z>(black_box(&input), black_box(&mut output))); }
        });
    }

    #[bench]
    fn bench_b32enc_simd(b: &mut Bencher) {
        let input = [0; 40];
        let mut output = [0u8; 64];
        b.iter(|| {
            unsafe { black_box(b32enc_simd::<Z>(black_box(&input), black_box(&mut output))); }
        });
    }

    #[bench]
    fn bench_b32enc(b: &mut Bencher) {
        let input = [0; 35];
        let mut output = [0u8; 56];
        b.iter(|| {
            black_box(b32enc(black_box(&input), black_box(&mut output), Z))
        });
    }
}
