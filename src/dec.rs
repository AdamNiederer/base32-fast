use std::arch::x86_64::*;
use std::mem::transmute;
use std::simd::{Simd};
use std::simd::cmp::{SimdPartialOrd};

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

unsafe fn from_char_avx512<const A: u8>(src: __m512i) -> __m512i {
    let sentinel_vec = _mm512_set1_epi8(u8::MAX as i8);
    match A {
        Rfc4648 => {
            let v_uc_az = _mm512_sub_epi8(src, _mm512_set1_epi8(b'A' as i8));
            let v_lc_az = _mm512_sub_epi8(src, _mm512_set1_epi8(b'a' as i8));
            let v_27 = _mm512_sub_epi8(src, _mm512_set1_epi8((b'2' - 26) as i8));

            let mask_uc_az_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'A' as i8));
            let mask_uc_az_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'Z' as i8));
            let mask_uc_az = _kand_mask64(mask_uc_az_ge, mask_uc_az_le);

            let mask_lc_az_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'a' as i8));
            let mask_lc_az_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'z' as i8));
            let mask_lc_az = _kand_mask64(mask_lc_az_ge, mask_lc_az_le);

            let mask_27_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'2' as i8));
            let mask_27_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'7' as i8));
            let mask_27 = _kand_mask64(mask_27_ge, mask_27_le);

            let value = _mm512_mask_blend_epi8(mask_27, sentinel_vec, v_27);
            let value = _mm512_mask_blend_epi8(mask_uc_az, value, v_uc_az);
            _mm512_mask_blend_epi8(mask_lc_az, value, v_lc_az)
        }
        Rfc4648Hex => {
            let v_09 = _mm512_sub_epi8(src, _mm512_set1_epi8(b'0' as i8)); 
            let v_uc_av = _mm512_sub_epi8(src, _mm512_set1_epi8((b'A' - 10) as i8));
            let v_lc_av = _mm512_sub_epi8(src, _mm512_set1_epi8((b'a' - 10)as i8));

            let mask_09_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'0' as i8));
            let mask_09_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'9' as i8));
            let mask_09 = _kand_mask64(mask_09_ge, mask_09_le);

            let mask_uc_av_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'A' as i8));
            let mask_uc_av_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'V' as i8));
            let mask_uc_av = _kand_mask64(mask_uc_av_ge, mask_uc_av_le); 

            let mask_lc_av_ge = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(b'a' as i8));
            let mask_lc_av_le = _mm512_cmple_epu8_mask(src, _mm512_set1_epi8(b'v' as i8));
            let mask_lc_av = _kand_mask64(mask_lc_av_ge, mask_lc_av_le);

            let value = _mm512_mask_blend_epi8(mask_uc_av, sentinel_vec, v_uc_av);
            let value = _mm512_mask_blend_epi8(mask_lc_av, value, v_lc_av);
            _mm512_mask_blend_epi8(mask_09, value, v_09)
        }
        Crockford => {
            let lut_0_63 = _mm512_loadu_si512(CROCKFORD_LUT.as_ptr() as *const _);
            let lut_64_127 = _mm512_loadu_si512(CROCKFORD_LUT.as_ptr().offset(64) as *const _);
            let lut_128_191 = _mm512_loadu_si512(CROCKFORD_LUT.as_ptr().offset(128) as *const _);
            let lut_192_255 = _mm512_loadu_si512(CROCKFORD_LUT.as_ptr().offset(192) as *const _);

            let mask_ge_64 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(64u8 as i8));
            let mask_ge_128 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(128u8 as i8));
            let mask_ge_192 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(192u8 as i8));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;
            
            let v_0_63 = _mm512_permutexvar_epi8(src, lut_0_63);
            let v_64_127 = _mm512_permutexvar_epi8(src, lut_64_127);
            let v_128_191 = _mm512_permutexvar_epi8(src, lut_128_191);
            let v_192_255 = _mm512_permutexvar_epi8(src, lut_192_255);

            let b_0_63 = _mm512_mask_blend_epi8(mask_0_63, src, v_0_63);
            let b_64_127 = _mm512_mask_blend_epi8(mask_64_127, b_0_63, v_64_127);
            let b_128_191 = _mm512_mask_blend_epi8(mask_128_191, b_64_127, v_128_191);
            _mm512_mask_blend_epi8(mask_192_256, b_128_191, v_192_255)

        }
        Geohash => {
            let lut_0_63 = _mm512_loadu_si512(GEOHASH_LUT.as_ptr() as *const _);
            let lut_64_127 = _mm512_loadu_si512(GEOHASH_LUT.as_ptr().offset(64) as *const _);
            let lut_128_191 = _mm512_loadu_si512(GEOHASH_LUT.as_ptr().offset(128) as *const _);
            let lut_192_255 = _mm512_loadu_si512(GEOHASH_LUT.as_ptr().offset(192) as *const _);

            let mask_ge_64 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(64u8 as i8));
            let mask_ge_128 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(128u8 as i8));
            let mask_ge_192 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(192u8 as i8));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;
            
            let v_0_63 = _mm512_permutexvar_epi8(src, lut_0_63);
            let v_64_127 = _mm512_permutexvar_epi8(src, lut_64_127);
            let v_128_191 = _mm512_permutexvar_epi8(src, lut_128_191);
            let v_192_255 = _mm512_permutexvar_epi8(src, lut_192_255);

            let b_0_63 = _mm512_mask_blend_epi8(mask_0_63, src, v_0_63);
            let b_64_127 = _mm512_mask_blend_epi8(mask_64_127, b_0_63, v_64_127);
            let b_128_191 = _mm512_mask_blend_epi8(mask_128_191, b_64_127, v_128_191);
            _mm512_mask_blend_epi8(mask_192_256, b_128_191, v_192_255)
        }
        Z => {
            let lut_0_63 = _mm512_loadu_si512(Z_LUT.as_ptr() as *const _);
            let lut_64_127 = _mm512_loadu_si512(Z_LUT.as_ptr().offset(64) as *const _);
            let lut_128_191 = _mm512_loadu_si512(Z_LUT.as_ptr().offset(128) as *const _);
            let lut_192_255 = _mm512_loadu_si512(Z_LUT.as_ptr().offset(192) as *const _);

            let mask_ge_64 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(64u8 as i8));
            let mask_ge_128 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(128u8 as i8));
            let mask_ge_192 = _mm512_cmpge_epu8_mask(src, _mm512_set1_epi8(192u8 as i8));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;

            // TODO: Why does just passing src without the subtraction for 64-255 work here?
            let v_0_63 = _mm512_permutexvar_epi8(src, lut_0_63);
            let v_64_127 = _mm512_permutexvar_epi8(_mm512_sub_epi8(src, _mm512_set1_epi8(64)), lut_64_127);
            let v_128_191 = _mm512_permutexvar_epi8(_mm512_sub_epi8(src, _mm512_set1_epi8(128u8 as i8)), lut_128_191);
            let v_192_255 = _mm512_permutexvar_epi8(_mm512_sub_epi8(src, _mm512_set1_epi8(192u8 as i8)), lut_192_255);

            let b_0_63 = _mm512_mask_blend_epi8(mask_0_63, src, v_0_63);
            let b_64_127 = _mm512_mask_blend_epi8(mask_64_127, b_0_63, v_64_127);
            let b_128_191 = _mm512_mask_blend_epi8(mask_128_191, b_64_127, v_128_191);
            _mm512_mask_blend_epi8(mask_192_256, b_128_191, v_192_255)
        }
        _ => core::hint::unreachable_unchecked(),
    }
}

unsafe fn from_char_simd<const A: u8>(src: Simd<u8, 64>) -> Simd<u8, 64> {
    match A {
        Rfc4648 => {
            let v_uc_az = src - Simd::splat(b'A');
            let v_lc_az = src - Simd::splat(b'a');
            let v_27 = src - Simd::splat(b'2' - 26);
            
            let mask_uc_az = src.simd_ge(Simd::splat(b'A')) & src.simd_le(Simd::splat(b'Z'));
            let mask_lc_az = src.simd_ge(Simd::splat(b'a')) & src.simd_le(Simd::splat(b'z'));
            let mask_27 = src.simd_ge(Simd::splat(b'2')) & src.simd_le(Simd::splat(b'7'));

            let b_uc_az = mask_uc_az.select(v_uc_az, Simd::splat(u8::MAX));
            let b_lc_az = mask_lc_az.select(v_lc_az, b_uc_az);
            let v_all = mask_27.select(v_27, b_lc_az);
            v_all
        }
        Rfc4648Hex => {
            let v_uc_av = src - Simd::splat(b'A' - 10);
            let v_lc_av = src - Simd::splat(b'a' - 10);
            let v_09 = src - Simd::splat(b'0');
            
            let mask_uc_av = src.simd_ge(Simd::splat(b'A')) & src.simd_le(Simd::splat(b'V'));
            let mask_lc_av = src.simd_ge(Simd::splat(b'a')) & src.simd_le(Simd::splat(b'v'));
            let mask_27 = src.simd_ge(Simd::splat(b'0')) & src.simd_le(Simd::splat(b'9'));

            let b_uc_az = mask_uc_av.select(v_uc_av, Simd::splat(u8::MAX));
            let b_lc_az = mask_lc_av.select(v_lc_av, b_uc_az);
            let v_all = mask_27.select(v_09, b_lc_az);
            v_all
        }
        Crockford => {
            let lut_0_63 = transmute::<_, *const Simd<u8, 64>>(CROCKFORD_LUT.as_ptr().add(0)).read_unaligned();
            let lut_64_127 = transmute::<_, *const Simd<u8, 64>>(CROCKFORD_LUT.as_ptr().add(64)).read_unaligned();
            let lut_128_191 = transmute::<_, *const Simd<u8, 64>>(CROCKFORD_LUT.as_ptr().add(128)).read_unaligned();
            let lut_192_255 = transmute::<_, *const Simd<u8, 64>>(CROCKFORD_LUT.as_ptr().add(192)).read_unaligned();

            let mask_ge_64 = src.simd_ge(Simd::splat(64));
            let mask_ge_128 = src.simd_ge(Simd::splat(128));
            let mask_ge_192 = src.simd_ge(Simd::splat(192));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;
            
            let v_0_63 = lut_0_63.swizzle_dyn(src);
            let v_64_127 = lut_64_127.swizzle_dyn(src - Simd::splat(64));
            let v_128_191 = lut_128_191.swizzle_dyn(src - Simd::splat(128));
            let v_192_255 = lut_192_255.swizzle_dyn(src - Simd::splat(192));

            let b_0_63 = mask_0_63.select(v_0_63, src);
            let b_64_127 = mask_64_127.select(v_64_127, b_0_63);
            let b_128_191 = mask_128_191.select(v_128_191, b_64_127);
            mask_192_256.select(v_192_255, b_128_191)
        }
        Geohash => {
            let lut_0_63 = transmute::<_, *const Simd<u8, 64>>(GEOHASH_LUT.as_ptr().add(0)).read_unaligned();
            let lut_64_127 = transmute::<_, *const Simd<u8, 64>>(GEOHASH_LUT.as_ptr().add(64)).read_unaligned();
            let lut_128_191 = transmute::<_, *const Simd<u8, 64>>(GEOHASH_LUT.as_ptr().add(128)).read_unaligned();
            let lut_192_255 = transmute::<_, *const Simd<u8, 64>>(GEOHASH_LUT.as_ptr().add(192)).read_unaligned();

            let mask_ge_64 = src.simd_ge(Simd::splat(64));
            let mask_ge_128 = src.simd_ge(Simd::splat(128));
            let mask_ge_192 = src.simd_ge(Simd::splat(192));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;
            
            let v_0_63 = lut_0_63.swizzle_dyn(src);
            let v_64_127 = lut_64_127.swizzle_dyn(src - Simd::splat(64));
            let v_128_191 = lut_128_191.swizzle_dyn(src - Simd::splat(128));
            let v_192_255 = lut_192_255.swizzle_dyn(src - Simd::splat(192));

            let b_0_63 = mask_0_63.select(v_0_63, src);
            let b_64_127 = mask_64_127.select(v_64_127, b_0_63);
            let b_128_191 = mask_128_191.select(v_128_191, b_64_127);
            mask_192_256.select(v_192_255, b_128_191)
        }
        Z => {
            let lut_0_63 = transmute::<_, *const Simd<u8, 64>>(Z_LUT.as_ptr().add(0)).read_unaligned();
            let lut_64_127 = transmute::<_, *const Simd<u8, 64>>(Z_LUT.as_ptr().add(64)).read_unaligned();
            let lut_128_191 = transmute::<_, *const Simd<u8, 64>>(Z_LUT.as_ptr().add(128)).read_unaligned();
            let lut_192_255 = transmute::<_, *const Simd<u8, 64>>(Z_LUT.as_ptr().add(192)).read_unaligned();

            let mask_ge_64 = src.simd_ge(Simd::splat(64));
            let mask_ge_128 = src.simd_ge(Simd::splat(128));
            let mask_ge_192 = src.simd_ge(Simd::splat(192));

            let mask_0_63 = !mask_ge_64;
            let mask_64_127 = mask_ge_64 & !mask_ge_128;
            let mask_128_191 = mask_ge_128 & !mask_ge_192;
            let mask_192_256 = mask_ge_192;
            
            let v_0_63 = lut_0_63.swizzle_dyn(src);
            let v_64_127 = lut_64_127.swizzle_dyn(src - Simd::splat(64));
            let v_128_191 = lut_128_191.swizzle_dyn(src - Simd::splat(128));
            let v_192_255 = lut_192_255.swizzle_dyn(src - Simd::splat(192));

            let b_0_63 = mask_0_63.select(v_0_63, src);
            let b_64_127 = mask_64_127.select(v_64_127, b_0_63);
            let b_128_191 = mask_128_191.select(v_128_191, b_64_127);
            mask_192_256.select(v_192_255, b_128_191)
        }
        _ => core::hint::unreachable_unchecked(),
    }
}

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

pub fn b32dec(src: & [u8], dst: & mut [u8], alphabet: u8) {
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

pub unsafe fn b32dec_generic<'a, const A: u8>(src: &'a [u8], dst: &'a mut [u8]) {
    if src.len() >= 64 {
        b32dec_avx512::<A>(src, dst);
    }

    let src_tail = src.len() - src.len() % 64;
    let dst_tail = dst.len() - dst.len() % 40;
    for (i, chunk) in src[src_tail..].chunks_exact(8).enumerate() {
        let split0 = ('A' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 0] >= 'A' as u8) as u8);
        let split1 = ('A' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 1] >= 'A' as u8) as u8);
        let split2 = ('A' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 2] >= 'A' as u8) as u8);
        let split3 = ('A' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 3] >= 'A' as u8) as u8);
        let split4 = ('A' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 4] >= 'A' as u8) as u8);

        dst[dst_tail + 5 * i + 0] = ((chunk[0] << 3) | (chunk[1] >> 2)) - (48 + split0);
        dst[dst_tail + 5 * i + 1] = ((chunk[1] << 6) | (chunk[2] << 1) | (chunk[3] >> 4)) - (48 + split1);
        dst[dst_tail + 5 * i + 2] = (chunk[3] << 4) | (chunk[4] >> 1) - (48 + split2);
        dst[dst_tail + 5 * i + 3] = ((chunk[4] << 7) | (chunk[5] << 2) | (chunk[6] >> 3)) - (48 + split3);
        dst[dst_tail + 5 * i + 4] = ((chunk[6] << 5) | chunk[7]) - (48 + split4);
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    fn expected_from_char(src: u8, alphabet: &[u8; 32]) -> u8 {
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
}
