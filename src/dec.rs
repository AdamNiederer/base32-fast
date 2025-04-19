use std::arch::x86_64::*;

fn b32dec_avx512<'a>(src: &'a [u8], dst: &'a mut [u8]) {
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 64 {
        unsafe {
            let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const i32);
            let m = _mm512_cmpge_epi8_mask(s, _mm512_set1_epi8('A' as i8));
            let a = _mm512_mask_blend_epi8(m, _mm512_set1_epi8('0' as i8), _mm512_set1_epi8('A' as i8 - 10));
            let d = _mm512_sub_epi8(s, a);
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
            _mm512_mask_storeu_epi8(dst.as_ptr().add(dst_cur) as *mut i8, 0x000000FFFFFFFFFF, res)
        }
        src_cur += 64;
        dst_cur += 40;
    }
}

pub fn b32dec<'a>(src: &'a [u8], dst: &'a mut [u8], alphabet: u8) {
    if src.len() >= 64 {
        b32dec_avx512(src, dst);
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


