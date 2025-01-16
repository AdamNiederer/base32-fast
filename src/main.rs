#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![allow(non_upper_case_globals)]

use std::simd::{Simd};
use std::simd::cmp::SimdPartialOrd;
use std::mem::transmute;

// const ALPHA: &[u8] = b"0123456789abcdefghijklmnopqrstuv";

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

#[inline(always)]
fn b32enc_avx512<'a>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
    use core::arch::x86_64::*;
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 40 {
        unsafe { 
            let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const i32);
            let p = _mm512_permutexvar_epi8(__m512i::from(shuf), s);
            let multishift = _mm512_set_epi8(
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
                59, 54, 49, 44, 39, 34, 29, 24,
            );
            let d = _mm512_multishift_epi64_epi8(multishift, p);
            let d = _mm512_and_si512(d, _mm512_set1_epi8(0x1F));
            // eprintln!("{:#066b}", Simd::<u64, 8>::from(d)[0]);
            let db = _mm512_permutexvar_epi8(__m512i::from(endian64), d);
            let m1 = _mm512_cmplt_epi8_mask(db, _mm512_set1_epi8(10));
            //let s1 = _mm512_mask_set1_epi8(_mm512_setzero_si512(), m1, i8::MIN);
            let res = _mm512_or_si512(
                _mm512_maskz_add_epi8(m1, db, _mm512_set1_epi8('0' as i8)),
                _mm512_maskz_add_epi8(!m1, db, _mm512_set1_epi8('a' as i8 - 10)),
            );
            
            _mm512_storeu_si512(dst.as_ptr().add(dst_cur) as *mut __m512i, res);
        }
        src_cur += 40;
        dst_cur += 64;
    }
    return dst;
}


#[inline(always)]
fn b32enc_simd<'a>(src: &'a [u8], dst: &'a mut [u8]) -> &'a [u8] {
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

        let db = unsafe { transmute::<_, Simd<u8, 64>>(d) }.swizzle_dyn(endian64);
        let m1 = db.simd_lt(Simd::splat(10));
        let s1 = unsafe { transmute::<_, Simd<u8, 64>>(m1.to_int()) };
        let res = (s1 & (db + Simd::splat('0' as u8)))
            | (!s1 & (db + Simd::splat('a' as u8 - 10)));

        unsafe { 
            transmute::<_, *mut Simd<u8, 64>>(dst.as_ptr().add(dst_cur)).write_unaligned(res);
        }
        src_cur += 40;
        dst_cur += 64;
    }

    return dst;
}

fn b32enc<'a>(src: &'a [u8], dst: &'a mut [u8]) {
    if src.len() >= 40 {
        b32enc_avx512(src, dst);        
    }

    let src_tail = src.len() - src.len() % 40;
    let dst_tail = dst.len() - dst.len() % 64;
    let residual = src.len() - src.len() % 5;
    for (i, chunk) in src[src_tail..residual].chunks_exact(5).enumerate() {
        dst[dst_tail + i * 8] = (chunk[0] & 0xf8) >> 3;
        dst[dst_tail + i * 8 + 1] = ((chunk[0] & 0x07) << 2) | ((chunk[1] & 0xC0) >> 6);
        dst[dst_tail + i * 8 + 2] = (chunk[1] & 0x3E) >> 1;
        dst[dst_tail + i * 8 + 3] = ((chunk[1] & 0x01) << 4) | ((chunk[2] & 0xF0) >> 4);
        dst[dst_tail + i * 8 + 4] = ((chunk[2] & 0x0F) << 1) | (chunk[3] >> 7);
        dst[dst_tail + i * 8 + 5] = (chunk[3] & 0x7C) >> 2;
        dst[dst_tail + i * 8 + 6] = ((chunk[3] & 0x03) << 3) | ((chunk[4] & 0xE0) >> 5);
        dst[dst_tail + i * 8 + 7] = chunk[4] & 0x1F;

        for j in 0..8 {
            let split = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 8 + j] >= 10) as u8);
            dst[dst_tail + i * 8 + j] += 48 + split;
        }
    }
}

fn b32dec_avx512<'a>(src: &'a [u8], dst: &'a mut [u8]) {
    use core::arch::x86_64::*;
    let mut src_cur = 0;
    let mut dst_cur = 0;
    while src.len() - src_cur >= 64 {
        unsafe {
            let s = _mm512_loadu_si512(src.as_ptr().add(src_cur) as *const i32);
            let m = _mm512_cmplt_epi8_mask(s, _mm512_set1_epi8('a' as i8));
            let d = _mm512_or_si512(
                _mm512_maskz_sub_epi8(m, s, _mm512_set1_epi8('0' as i8)),
                _mm512_maskz_sub_epi8(!m, s, _mm512_set1_epi8('a' as i8 - 10)),
            );
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
            let s32s = _mm512_shuffle_epi32(s32, 0b10_11_00_01);
            // -> 000000000000aaaaaaaabbbbbbbbcccc 000000000000ccccddddddddeeeeeeee
            let s32ss = _mm512_srl_epi64(s32s, _mm_set1_epi64x(12));
            // -> 000000000000000000000000aaaaaaaa bbbbbbbbcccc000000000000ccccdddd
            
            let mask = _mm512_set1_epi64(0b0000000000001111111111111111111111111111111100000000000000000000);
            // select bits from s32s if mask = 1, else s32ss
            let blitted = _mm512_ternarylogic_epi64(mask, s32ss, s32s, 0xca); 
            
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
            // eprintln!("{:?}", Simd::<u8, 64>::from(res));
            // eprint!("{:?} ", dst_cur);

            _mm512_mask_storeu_epi8(dst.as_ptr().add(dst_cur) as *mut i8, 0x000000FFFFFFFFFF, res)
        }
        src_cur += 64;
        dst_cur += 40;
    }
}

fn b32dec<'a>(src: &'a [u8], dst: &'a mut [u8]) {
    if src.len() >= 64 {
        b32dec_avx512(src, dst);        
    }

    let src_tail = src.len() - src.len() % 64;
    let dst_tail = dst.len() - dst.len() % 40;
    for (i, chunk) in src[src_tail..].chunks_exact(8).enumerate() {
        let split0 = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 0] >= 'a' as u8) as u8);
        let split1 = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 1] >= 'a' as u8) as u8);
        let split2 = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 2] >= 'a' as u8) as u8);
        let split3 = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 3] >= 'a' as u8) as u8);
        let split4 = ('a' as u8 - '0' as u8 - 10) * ((dst[dst_tail + i * 5 + 4] >= 'a' as u8) as u8);

        dst[dst_tail + 5 * i + 0] = ((chunk[0] << 3) | (chunk[1] >> 2)) - (48 + split0);
        dst[dst_tail + 5 * i + 1] = ((chunk[1] << 6) | (chunk[2] << 1) | (chunk[3] >> 4)) - (48 + split1);
        dst[dst_tail + 5 * i + 2] = (chunk[3] << 4) | (chunk[4] >> 1) - (48 + split2);
        dst[dst_tail + 5 * i + 3] = ((chunk[4] << 7) | (chunk[5] << 2) | (chunk[6] >> 3)) - (48 + split3);
        dst[dst_tail + 5 * i + 4] = ((chunk[6] << 5) | chunk[7]) - (48 + split4);
}
}


use clap::Parser;
use std::io;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::fs::File;
#[derive(Parser, Debug)]
struct Args {
    #[arg()]
    input: Option<String>,
        
    #[arg(short, long)]
    decode: bool,

    #[arg(short, long)]
    encode: bool,
}

fn main() {
    use core::arch::x86_64::*;
    // unsafe { 
    //     let a = _mm512_set_epi8(
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, 0, 0,
    //         0, 0, 0, 0, 0, 0, -1, 0,
    //     );
    //     let b = _mm512_set_epi64(1, 1, 1, 1, 1, 1, 1, 1);
    //     let d = _mm512_multishift_epi64_epi8(a, b);
    //     eprintln!("{:#018x}", Simd::<u64, 8>::from(d)[0]);
    // }
    // return;

    // let src = (1..65).collect::<Vec<u8>>();
    // let src2 = b"1234567890qwertyuiopasdfghjklzxcvbnm-=_+";
    // let src3 = [
    //     0x01, 0xFF, 0x02, 0xFF,
    //     0x03, 0xFF, 0x04, 0xFF,
    //     0x05, 0xFF, 0x06, 0xFF,
    //     0x07, 0xFF, 0x08, 0xFF,
    //     0x09, 0xFF, 0x0A, 0xFF,
    //     0x0B, 0xFF, 0x0C, 0xFF,
    //     0x0D, 0xFF, 0x0E, 0xFF,
    //     0x0F, 0xFF, 0x10, 0xFF,
    //     0x11, 0xFF, 0x12, 0xFF,
    //     0x13, 0xFF, 0x14, 0xFF,
    // ] as [u8; 40];
    // let srcv = unsafe { _mm256_loadu_si256(src3.as_ptr() as *const __m256i) };
    // eprintln!("{:#066b}", Simd::<u64, 4>::from(srcv)[0]);
    
    // let mut dst = [0x00u8; 64];
    // let mut ddst = [0x00u8; 40];
    // b32enc(src2, &mut dst);
    // b32dec(&dst, &mut ddst);
    // println!("{}", String::from_utf8_lossy(&dst));
    // println!("{}", String::from_utf8_lossy(&ddst));
    // return;

    
    let args = Args::parse();

    let mut reader = {
        if let Some(ref path) = args.input {
            Box::new(BufReader::new(match File::open(path) {
                Ok(path) => path,
                _ => {
                    writeln!(io::stderr(), "base32: no such file: {}", path).expect("base32: stderr write error");
                    return;
                }
            })) as Box<dyn BufRead>
        } else {
            Box::new(BufReader::new(io::stdin())) as Box<dyn BufRead>
        }
    };

    let mut writer = BufWriter::new(io::stdout());

    if args.decode {
        let mut write_buf = [0u8; 10000];
        let mut read_buf = [0u8; 16000];
        let mut residual = 0;
        let mut residual_buf = [0; 8];

        while let Ok(num_read) = reader.read(&mut read_buf[residual..]) {
            if num_read == 0 {
                break; // finalize
            }
            let len = residual + num_read;
            let tail = len / 8 * 8;
            let expected_out = tail / 8 * 5;
            residual = len - tail;

            b32dec(&read_buf[..tail], &mut write_buf[..expected_out]);
            residual_buf[..residual].copy_from_slice(&read_buf[tail..len]);
            read_buf[..residual].copy_from_slice(&residual_buf[..residual]);

            match writer.write_all(&mut write_buf[..expected_out]) {
                Ok(_) => (),
                _ => {
                    writeln!(io::stderr(), "base32: write error").expect("base32: stderr write error");
                    return;
                }
            }
        }
        writer.flush().expect("Write error");
    } else { 
        let mut write_buf = [0u8; 16000];
        let mut read_buf = [0u8; 10000];
        let mut residual = 0;
        let mut residual_buf = [0; 5];
        while let Ok(num_read) = reader.read(&mut read_buf[residual..]) {
            if num_read == 0 {
                break; // finalize
            }

            let len = residual + num_read;
            let tail = len / 5 * 5;
            let expected_out = tail / 5 * 8;
            residual = len - tail;
            b32enc(&read_buf[..tail], &mut write_buf[..expected_out]);
            residual_buf[..residual].copy_from_slice(&read_buf[tail..len]);
            read_buf[..residual].copy_from_slice(&residual_buf[..residual]);
            match writer.write_all(&mut write_buf[..expected_out]) {
                Ok(_) => (),
                _ => {
                    writeln!(io::stderr(), "base32: write error").expect("base32: stderr write error");
                    return;
                }
            }
        }
        writer.flush().expect("Write error");
    }
}

