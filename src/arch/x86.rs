// Copyright 2025 Don MacAskill. Licensed under MIT or Apache-2.0.

//! This module provides x86-specific implementations of the ArchOps trait.
//!
//! This module is designed to work with both x86 and x86_64 architectures.
//!
//! It uses the SSE2 and SSE4.1 instruction sets for SIMD operations.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use std::is_x86_feature_detected;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::traits::ArchOps;

macro_rules! make_shift {
    ($name:ident, $intrinsic:ident, $imm:expr) => {
        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn $name(&self, v: __m128i) -> __m128i {
            $intrinsic(v, $imm)
        }
    };
}

#[derive(Debug, Copy, Clone)]
pub struct X86Ops;

impl ArchOps for X86Ops {
    type Vector = __m128i;

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn create_vector_from_u64_pair(
        &self,
        high: u64,
        low: u64,
        reflected: bool,
    ) -> Self::Vector {
        // Note order is different from AArch64
        if reflected {
            Self::set_epi64x(low, high)
        } else {
            Self::set_epi64x(high, low)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn create_vector_from_u64_pair_non_reflected(
        &self,
        high: u64,
        low: u64,
    ) -> Self::Vector {
        // Note order is different from AArch64
        Self::set_epi64x(high, low)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn create_vector_from_u64(&self, value: u64, high: bool) -> Self::Vector {
        // x86 uses custom helper
        Self::create_u64_vector(value, high)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_u64s(&self, vector: Self::Vector) -> [u64; 2] {
        [
            Self::extract_u64_low(vector),
            Self::extract_u64_high(vector),
        ]
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_poly64s(&self, vector: Self::Vector) -> [u64; 2] {
        // On x86, poly64s and u64s extraction is the same
        self.extract_u64s(vector)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn xor_vectors(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm_xor_si128(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_bytes(&self, ptr: *const u8) -> Self::Vector {
        // x86 requires cast to __m128i*
        _mm_loadu_si128(ptr as *const __m128i)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_aligned(&self, ptr: *const [u64; 2]) -> Self::Vector {
        // x86 requires cast to __m128i*
        _mm_loadu_si128(ptr as *const __m128i)
    }

    #[inline]
    unsafe fn shuffle_bytes(&self, data: Self::Vector, mask: Self::Vector) -> Self::Vector {
        // x86 uses specific SSSE3 instruction
        if is_x86_feature_detected!("ssse3") {
            Self::shuffle_bytes_ssse3(data, mask)
        } else {
            Self::shuffle_bytes_fallback(data, mask)
        }
    }

    #[inline]
    unsafe fn blend_vectors(
        &self,
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Vector,
    ) -> Self::Vector {
        // x86 has native blend that uses MSB automatically
        if is_x86_feature_detected!("sse4.1") {
            Self::blend_vectors_sse41(a, b, mask)
        } else if is_x86_feature_detected!("sse2") {
            Self::blend_vectors_sse2(a, b, mask)
        } else {
            Self::blend_vectors_fallback(a, b, mask)
        }
    }

    make_shift!(shift_left_4, _mm_slli_si128, 4);
    make_shift!(shift_left_8, _mm_slli_si128, 8);
    make_shift!(shift_left_12, _mm_slli_si128, 12);
    make_shift!(shift_left_32, _mm_slli_si128, 4); // 4-byte == 32‑bit

    make_shift!(shift_right_4, _mm_srli_si128, 4);
    make_shift!(shift_right_5, _mm_srli_si128, 5);
    make_shift!(shift_right_6, _mm_srli_si128, 6);
    make_shift!(shift_right_7, _mm_srli_si128, 7);
    make_shift!(shift_right_8, _mm_srli_si128, 8);
    make_shift!(shift_right_12, _mm_srli_si128, 12);
    make_shift!(shift_right_32, _mm_srli_si128, 4); // 4-byte == 32‑bit

    #[inline]
    unsafe fn create_vector_from_u32(&self, value: u32, high: bool) -> Self::Vector {
        if is_x86_feature_detected!("sse4.1") {
            Self::create_vector_from_u32_sse41(value, high)
        } else if is_x86_feature_detected!("sse2") {
            Self::create_vector_from_u32_sse2(value, high)
        } else {
            Self::create_vector_from_u32_fallback(value, high)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set_all_bytes(&self, value: u8) -> Self::Vector {
        _mm_set1_epi8(value as i8)
    }

    #[inline(always)]
    unsafe fn create_compare_mask(&self, vector: Self::Vector) -> Self::Vector {
        // On x86, MSB is already used for blending, so we just return the vector
        vector
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn and_vectors(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        _mm_and_si128(a, b)
    }

    #[inline]
    unsafe fn carryless_mul_00(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if is_x86_feature_detected!("pclmulqdq") {
            Self::carryless_mul_00_hw(a, b)
        } else if is_x86_feature_detected!("sse2") {
            Self::carryless_mul_sse2(a, b, 0x00)
        } else {
            Self::carryless_mul_fallback(a, b, 0x00)
        }
    }
    #[inline]
    unsafe fn carryless_mul_01(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if is_x86_feature_detected!("pclmulqdq") {
            Self::carryless_mul_01_hw(a, b)
        } else if is_x86_feature_detected!("sse2") {
            Self::carryless_mul_sse2(a, b, 0x01)
        } else {
            Self::carryless_mul_fallback(a, b, 0x01)
        }
    }
    #[inline]
    unsafe fn carryless_mul_10(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if is_x86_feature_detected!("pclmulqdq") {
            Self::carryless_mul_10_hw(a, b)
        } else if is_x86_feature_detected!("sse2") {
            Self::carryless_mul_sse2(a, b, 0x10)
        } else {
            Self::carryless_mul_fallback(a, b, 0x10)
        }
    }
    #[inline]
    unsafe fn carryless_mul_11(&self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        if is_x86_feature_detected!("pclmulqdq") {
            Self::carryless_mul_11_hw(a, b)
        } else if is_x86_feature_detected!("sse2") {
            Self::carryless_mul_sse2(a, b, 0x11)
        } else {
            Self::carryless_mul_fallback(a, b, 0x11)
        }
    }
}

impl X86Ops {
    // Helper methods specific to x86/x86_64
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn set_epi64x(high: u64, low: u64) -> __m128i {
        #[cfg(target_arch = "x86_64")]
        {
            _mm_set_epi64x(high as i64, low as i64)
        }
        #[cfg(target_arch = "x86")]
        {
            // _mm_set_epi32 takes (highest, higher, lower, lowest)
            // We need to ensure e0 is in the lower 64 bits and e1 in the higher 64 bits
            let low_vec_part = _mm_set_epi32(0, 0, (low >> 32) as i32, (low & 0xFFFFFFFF) as i32);
            let high_vec_part =
                _mm_set_epi32(0, 0, (high >> 32) as i32, (high & 0xFFFFFFFF) as i32);
            _mm_unpacklo_epi64(low_vec_part, high_vec_part)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn create_u64_vector(value: u64, high: bool) -> __m128i {
        if high {
            Self::set_epi64x(value, 0)
        } else {
            Self::set_epi64x(0, value)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_u64_low(v: __m128i) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            _mm_cvtsi128_si64(v) as u64
        }
        #[cfg(target_arch = "x86")]
        {
            let lo32 = _mm_cvtsi128_si32(v) as u32 as u64;
            let hi32 = _mm_cvtsi128_si32(_mm_srli_si128(v, 4)) as u32 as u64;
            (hi32 << 32) | lo32
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_u64_high(v: __m128i) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            _mm_cvtsi128_si64(_mm_srli_si128(v, 8)) as u64
        }
        #[cfg(target_arch = "x86")]
        {
            let lo32 = _mm_cvtsi128_si32(_mm_srli_si128(v, 8)) as u32 as u64;
            let hi32 = _mm_cvtsi128_si32(_mm_srli_si128(v, 12)) as u32 as u64;
            (hi32 << 32) | lo32
        }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn create_vector_from_u32_sse41(value: u32, high: bool) -> __m128i {
        if high {
            _mm_insert_epi32(_mm_setzero_si128(), value as i32, 3)
        } else {
            _mm_set_epi32(0, 0, 0, value as i32)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn create_vector_from_u32_sse2(value: u32, high: bool) -> __m128i {
        let v = _mm_cvtsi32_si128(value as i32);
        if high {
            _mm_shuffle_epi32(v, 0x15)
        } else {
            v
        }
    }

    #[inline]
    unsafe fn create_vector_from_u32_fallback(value: u32, high: bool) -> __m128i {
        let mut lanes = [0u32; 4];
        if high {
            lanes[3] = value;
        } else {
            lanes[0] = value;
        }
        core::mem::transmute(lanes)
    }

    #[target_feature(enable = "ssse3")]
    unsafe fn shuffle_bytes_ssse3(data: __m128i, mask: __m128i) -> __m128i {
        _mm_shuffle_epi8(data, mask)
    }

    /// Fallback for `_mm_shuffle_epi8(a, mask)`.
    #[inline]
    unsafe fn shuffle_bytes_fallback(data: __m128i, mask: __m128i) -> __m128i {
        let bytes_a: [u8; 16] = core::mem::transmute(data);
        let bytes_m: [u8; 16] = core::mem::transmute(mask);

        let mut out = [0u8; 16];
        for i in 0..16 {
            let ctl = bytes_m[i];
            if (ctl & 0x80) == 0 {
                out[i] = bytes_a[(ctl & 0x0F) as usize];
            }
        }

        core::mem::transmute(out)
    }

    #[target_feature(enable = "sse2,pclmulqdq")]
    unsafe fn carryless_mul_00_hw(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x00)
    }
    #[target_feature(enable = "sse2,pclmulqdq")]
    unsafe fn carryless_mul_01_hw(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x01)
    }
    #[target_feature(enable = "sse2,pclmulqdq")]
    unsafe fn carryless_mul_10_hw(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x10)
    }
    #[target_feature(enable = "sse2,pclmulqdq")]
    unsafe fn carryless_mul_11_hw(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x11)
    }

    /// Implementation of _mm_clmulepi64_si128 without intrinsics
    #[inline]
    unsafe fn carryless_mul_fallback(a: __m128i, b: __m128i, imm: u8) -> __m128i {
        // Extract the __m128i values as arrays of u64
        let a_parts: [u64; 2] = std::mem::transmute(a);
        let b_parts: [u64; 2] = std::mem::transmute(b);

        // Select which 64-bit parts to multiply based on immediate value
        let (a_val, b_val) = match imm & 0b0001_0001 {
            0b0000_0000 => (a_parts[0], b_parts[0]), // low64(a) * low64(b)
            0b0000_0001 => (a_parts[1], b_parts[0]), // high64(a) * low64(b)
            0b0001_0000 => (a_parts[0], b_parts[1]), // low64(a) * high64(b)
            0b0001_0001 => (a_parts[1], b_parts[1]), // high64(a) * high64(b)
            _ => panic!("invalid imm in call to carryless_mul_sw"),
        };

        // Perform carryless multiplication
        let result = Self::carryless_multiply_64(a_val.into(), b_val.into());

        // Convert back to __m128i
        std::mem::transmute(result)
    }

    /// Performs carryless multiplication of two 128-bit values
    #[inline]
    fn carryless_multiply_64(mut a: u128, mut b: u128) -> u128 {
        let mut res: u128 = 0;
        while b != 0 {
            if b & 1 != 0 {
                res ^= a; // XOR, not addition
            }
            a <<= 1; // multiply by x in GF(2)[x]
            b >>= 1;
        }
        res
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn carryless_mul_sse2(a: __m128i, b: __m128i, imm: u8) -> __m128i {
        let a_sel = if (imm & 0x01) != 0 {
            _mm_srli_si128(a, 8)
        } else {
            a
        };
        let b_sel = if (imm & 0x10) != 0 {
            _mm_srli_si128(b, 8)
        } else {
            b
        };

        #[cfg(target_arch = "x86_64")]
        let a_val: u64 = _mm_cvtsi128_si64(a_sel) as u64;
        #[cfg(target_arch = "x86")]
        let a_val: u64 = {
            let mut tmp: u64 = 0;
            _mm_storel_epi64(&mut tmp as *mut u64 as *mut __m128i, a_sel);
            tmp
        };

        #[cfg(target_arch = "x86_64")]
        let b_val: u64 = _mm_cvtsi128_si64(b_sel) as u64;
        #[cfg(target_arch = "x86")]
        let b_val: u64 = {
            let mut tmp: u64 = 0;
            _mm_storel_epi64(&mut tmp as *mut u64 as *mut __m128i, b_sel);
            tmp
        };

        let mut x = a_val as u128;
        let mut y = b_val as u128;
        let mut prod: u128 = 0;
        while y != 0 {
            if (y & 1) != 0 {
                prod ^= x;
            }
            x <<= 1;
            y >>= 1;
        }

        let lo = prod as i64;
        let hi = (prod >> 64) as i64;
        _mm_set_epi64x(hi, lo)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn blend_vectors_sse41(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
        _mm_blendv_epi8(a, b, mask)
    }

    #[inline]
    unsafe fn blend_vectors_fallback(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
        let bytes_a: [u8; 16] = core::mem::transmute(a);
        let bytes_b: [u8; 16] = core::mem::transmute(b);
        let bytes_mask: [u8; 16] = core::mem::transmute(mask);

        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if (bytes_mask[i] & 0x80) != 0 {
                bytes_b[i]
            } else {
                bytes_a[i]
            };
        }

        core::mem::transmute(out)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn blend_vectors_sse2(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
        let zero = _mm_setzero_si128();
        let sel = _mm_cmplt_epi8(mask, zero);
        let a_sel = _mm_andnot_si128(sel, a);
        let b_sel = _mm_and_si128(b, sel);
        _mm_or_si128(a_sel, b_sel)
    }
}
