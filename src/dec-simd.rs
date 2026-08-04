use std::mem::transmute;
use std::simd::{Simd, Select};
use std::simd::cmp::SimdPartialOrd;

use crate::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z};

pub(crate) unsafe fn from_char_simd<const A: u8>(src: Simd<u8, 64>) -> Simd<u8, 64> {
    let lut = match A {
        Rfc4648 => &crate::dec::RFC4648_LUT,
        Rfc4648Hex => &crate::dec::RFC4648HEX_LUT,
        Crockford => &crate::dec::CROCKFORD_LUT,
        Geohash => &crate::dec::GEOHASH_LUT,
        Z => &crate::dec::Z_LUT,
        _ => core::hint::unreachable_unchecked(),
    };

    let lut_0_63 = transmute::<_, *const Simd<u8, 64>>(lut.as_ptr().add(0)).read_unaligned();
    let lut_64_127 = transmute::<_, *const Simd<u8, 64>>(lut.as_ptr().add(64)).read_unaligned();
    let mask_ge_64 = src.simd_ge(Simd::splat(64));

    let v_0_63 = lut_0_63.swizzle_dyn(src);
    let v_64_127 = lut_64_127.swizzle_dyn(src & Simd::splat(0x3F));

    mask_ge_64.select(v_64_127, v_0_63)
}

#[inline(never)]
pub(crate) unsafe fn b32dec_simd<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) {
    let mut src_cur = 0;
    let mut dst_cur = 0;

    let byte_mask = Simd::<u64, 8>::splat(0xFF);

    let pack_idx = Simd::<u8, 64>::from_array([
        0, 1, 2, 3, 4, 8, 9, 10, 11, 12,
        16, 17, 18, 19, 20, 24, 25, 26, 27, 28,
        32, 33, 34, 35, 36, 40, 41, 42, 43, 44,
        48, 49, 50, 51, 52, 56, 57, 58, 59, 60,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ]);

    while src.len() - src_cur >= 64 {
        let s = Simd::<u8, 64>::from_slice(&src[src_cur..src_cur + 64]);
        let d = from_char_simd::<A>(s);
        let g: Simd<u64, 8> = transmute(d);

        let g0 = g & byte_mask;
        let g1 = (g >> 8) & byte_mask;
        let g2 = (g >> 16) & byte_mask;
        let g3 = (g >> 24) & byte_mask;
        let g4 = (g >> 32) & byte_mask;
        let g5 = (g >> 40) & byte_mask;
        let g6 = (g >> 48) & byte_mask;
        let g7 = g >> 56;

        let o0 = (g0 << 3) | (g1 >> 2);
        let o1 = ((g1 << 6) | (g2 << 1) | (g3 >> 4)) & byte_mask;
        let o2 = ((g3 << 4) | (g4 >> 1)) & byte_mask;
        let o3 = ((g4 << 7) | (g5 << 2) | (g6 >> 3)) & byte_mask;
        let o4 = ((g6 << 5) | g7) & byte_mask;

        let out64 = o0 | (o1 << 8) | (o2 << 16) | (o3 << 24) | (o4 << 32);

        let out_bytes: Simd<u8, 64> = transmute(out64);
        let packed = out_bytes.swizzle_dyn(pack_idx);
        core::ptr::copy_nonoverlapping(packed.as_array().as_ptr(), dst.as_mut_ptr().add(dst_cur), 40);

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
    use base32::{Alphabet, encode, decode};

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
    fn test_b32dec_simd_rfc4648() {
        let input = b"ORSXG5DJORUXG5LNORUWYZLSEBFWC2LTN5ZG64DDMNWGC2LPN5ZG64TON5XHIZLE";
        let expected = decode(Alphabet::Rfc4648 { padding: true }, core::str::from_utf8(input).unwrap()).unwrap();
        let mut output = [0u8; 40];
        unsafe {
            b32dec_simd::<{Rfc4648}>(input, &mut output);
        }
        assert_eq!(&output[..expected.len()], &expected);
    }

    #[test]
    fn test_b32dec_simd_boundary() {
        for data_len in [1, 2, 3, 4, 5, 6, 7, 63, 64, 65, 127, 128, 129] {
            let data: Vec<u8> = (0..data_len).map(|i| i as u8).collect();
            let encoded = encode(Alphabet::Rfc4648 { padding: false }, &data);
            let expected = decode(Alphabet::Rfc4648 { padding: false }, &encoded).unwrap();
            let mut dst = vec![0u8; (expected.len() + 4) / 5 * 5];
            let dst = crate::dec::b32dec(encoded.as_bytes(), &mut dst, Rfc4648);
            assert_eq!(dst, expected, "failed for length {}", data_len);
            let encoded_pad = encode(Alphabet::Rfc4648 { padding: true }, &data);
            let expected_pad = decode(Alphabet::Rfc4648 { padding: true }, &encoded_pad).unwrap();
            let mut dst_pad = vec![0u8; (expected_pad.len() + 4) / 5 * 5];
            let dst_pad = crate::dec::b32dec(encoded_pad.as_bytes(), &mut dst_pad, Rfc4648);
            assert_eq!(dst_pad, expected_pad, "failed for padded length {}", data_len);
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
    fn bench_b32dec_simd(b: &mut Bencher) {
        let input = b"GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";
        let mut output = [0u8; 40];
        b.iter(|| {
            unsafe { black_box(b32dec_simd::<Z>(black_box(input), black_box(&mut output))) };
        });
    }

    #[bench]
    fn bench_b32dec_simd_bulk(b: &mut Bencher) {
        let mut input = vec![0u8; 16777216];
        let mut output = vec![0u8; 10485760];
        for (i, b) in input.iter_mut().enumerate() {
            *b = b"GEZDGNBVGY3TQOJQ"[i % 16];
        }
        b.iter(|| {
            unsafe { black_box(b32dec_simd::<Z>(black_box(&input), black_box(&mut output))) };
        });
    }
}
