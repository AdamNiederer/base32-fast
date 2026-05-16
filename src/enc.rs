use crate::{
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
pub(crate) unsafe fn to_char<const A: u8>(value: u8) -> u8 {
    match A {
        Rfc4648 => RFC4648_CHARS[value as usize],
        Rfc4648Hex => RFC4648HEX_CHARS[value as usize],
        Crockford => CROCKFORD_CHARS[value as usize],
        Geohash => GEOHASH_CHARS[value as usize],
        Z => Z_CHARS[value as usize],
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

#[path = "enc-scalar.rs"]
mod enc_scalar;
use enc_scalar::b32enc_generic;

#[cfg(feature = "simd")]
#[path = "enc-simd.rs"]
mod enc_simd;

#[cfg(feature = "avx-512")]
#[path = "enc-avx512.rs"]
mod enc_avx512;

#[cfg(test)]
mod tests {
    use super::*;
    use base32::{encode, Alphabet};

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
}
