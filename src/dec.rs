use crate::{
    Rfc4648, Rfc4648Hex, Crockford, Geohash, Z,
    RFC4648_CHARS, RFC4648HEX_CHARS, CROCKFORD_CHARS, GEOHASH_CHARS, Z_CHARS,
};

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

pub(crate) const RFC4648_LUT: [u8; 256] = generate_decode_lut(RFC4648_CHARS);
pub(crate) const RFC4648HEX_LUT: [u8; 256] = generate_decode_lut(RFC4648HEX_CHARS);
pub(crate) const CROCKFORD_LUT: [u8; 256] = generate_decode_lut(CROCKFORD_CHARS);
pub(crate) const GEOHASH_LUT: [u8; 256] = generate_decode_lut(GEOHASH_CHARS);
pub(crate) const Z_LUT: [u8; 256] = generate_decode_lut(Z_CHARS);

#[inline(always)]
pub(crate) unsafe fn from_char<const A: u8>(value: u8) -> u8 {
    match A {
        Rfc4648 => RFC4648_LUT[value as usize],
        Rfc4648Hex => RFC4648HEX_LUT[value as usize],
        Crockford => CROCKFORD_LUT[value as usize],
        Geohash => GEOHASH_LUT[value as usize],
        Z => Z_LUT[value as usize],
        _ => core::hint::unreachable_unchecked(),
    }
}

pub fn b32dec<'a>(src: &'a [u8], dst: &'a mut [u8], alphabet: u8) -> &'a [u8] {
    if src.len() == 0 {
        return &dst[0..0];
    }

    if dst.len() < ((src.len() + 7) / 8) * 5 {
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

#[path = "dec-scalar.rs"]
mod dec_scalar;
use dec_scalar::b32dec_generic;

#[cfg(feature = "simd")]
#[path = "dec-simd.rs"]
mod dec_simd;

#[cfg(feature = "avx-512")]
#[path = "dec-avx512.rs"]
mod dec_avx512;

#[cfg(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use base32::{Alphabet, encode, decode};
    use test::Bencher;

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

        for data_len in 60..70 {
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

        for data_len in 60..70 {
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
    fn test_b32dec_underalloced_dst() {
        let cases = [(2, 5), (4, 5), (5, 5), (7, 5), (8, 5), (10, 10), (12, 10), (13, 10), (15, 10), (16, 10)];
        for &(encoded_len, needed) in &cases {
            let encoded: Vec<u8> = (0..encoded_len)
                .map(|i| b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"[i % 32])
                .collect();
            let mut dst = vec![0u8; needed - 1];
            assert!(catch_unwind(AssertUnwindSafe(|| { b32dec(&encoded, &mut dst, Rfc4648); })).is_err());
            let mut dst = vec![0u8; needed];
            assert!(catch_unwind(AssertUnwindSafe(|| {b32dec(&encoded, &mut dst, Rfc4648); })).is_ok());
        }
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

        let expected = decode(Alphabet::Crockford, encoded_base).unwrap();

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

    #[bench]
    fn bench_base32_decode(b: &mut Bencher) {
        let input = std::str::from_utf8(&[b'A'; 64]).unwrap();
        b.iter(|| {
            black_box(decode(Alphabet::Rfc4648 { padding: true }, black_box(input)).unwrap());
        });
    }
}
