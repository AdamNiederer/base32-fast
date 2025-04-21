#![cfg_attr(test, feature(test))]

#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

pub const Rfc4648: u8 = 0;
pub const Rfc4648Hex: u8 = 1;
pub const Crockford: u8 = 2;
pub const Geohash: u8 = 3;
pub const Z: u8 = 4;

pub(crate) const RFC4648_CHARS: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
pub(crate) const RFC4648HEX_CHARS: &[u8; 32] = b"0123456789ABCDEFGHIJKLMNOPQRSTUV";
pub(crate) const CROCKFORD_CHARS: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
pub(crate) const GEOHASH_CHARS: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
pub(crate) const Z_CHARS: &[u8; 32] = b"ybndrfg8ejkmcpqxot1uwisza345h769";

mod enc;
mod dec;

pub use crate::enc::{b32enc};
pub use crate::dec::{b32dec};
