#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![allow(non_upper_case_globals)]

use clap::Parser;
use std::io;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::fs::File;

use base32_fast::{Rfc4648, Rfc4648Hex, Crockford, Geohash, Z, b32enc, b32dec};

#[derive(Parser, Debug)]
struct Args {
    /// The file from which input will be read. If not provided, read from stdin.
    #[arg()]
    input: Option<String>,

    /// Whether to decode the input
    #[arg(short, long)]
    decode: bool,

    /// Whether to encode the input
    #[arg(short, long)]
    encode: bool,

    /// The alphabet to use (rfc4648, rfc4648hex, crockford, geohash, or z; default rfc4648)
    #[arg(long)]
    alphabet: Option<String>,
}

fn main() {
    let args = Args::parse();

    let alphabet = match args.alphabet.as_ref().map(|x| x.as_str()) {
        Some("rfc4648") => Rfc4648,
        Some("rfc4648hex") => Rfc4648Hex,
        Some("crockford") => Crockford,
        Some("geohash") => Geohash,
        Some("z") => Z,
        None => Rfc4648,
        Some(_) => panic!("unknown alphabet {}", args.alphabet.as_ref().unwrap().as_str()),
    };

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
        let mut write_buf = [0u8; 100000];
        let mut read_buf = [0u8; 160000];
        while let Ok(num_read) = reader.read(&mut read_buf) {
            if num_read == 0 {
                break;
            }
            let expected_out = num_read / 8 * 5;
            b32dec(&read_buf[..num_read], &mut write_buf[..expected_out], alphabet);
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
        let mut write_buf = [0u8; 160000];
        let mut read_buf = [0u8; 100000];
        while let Ok(num_read) = reader.read(&mut read_buf) {
            if num_read == 0 {
                break;
            }
            let expected_out = (num_read + 4) / 5 * 8;
            b32enc(&read_buf[..num_read], &mut write_buf[..expected_out], alphabet);
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
