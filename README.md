# base32-fast

An adaptation of Wojciech Mula's base64 encoding/decoding algorithms. Obscenely
fast, over 9GB/s on a good chip. Supports all of the major alphabets, but
Rfc4648 and Rfc4648Hex are faster by a hair because they don't need a shuffle.

Decoder is fully working but alphabets other than Rfc4648 are not yet done and
there are no tests for it.
