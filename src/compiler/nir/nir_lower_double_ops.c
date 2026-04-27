/*
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_double_pack_records.h"

#include <float.h>
#include <math.h>

/* tf96 softfp64 chain-optimization infrastructure.
 *
 * For fmul (and later fadd/ffma), we avoid re-doing the fold + split +
 * pack + unfold work on every op by keeping each fp64 SSA def's
 * "folded" form alive alongside a shift (int exponent offset from
 * 0x3FF) through the NIR graph. The triple-float is in a safe fp32
 * range; the shift tracks the fp64 exponent at the NIR int level and
 * only gets baked back into bits at the final pack.
 *
 * Cache keys `(src_ssa, swizzle_index)` because at this point 64-bit
 * ALU ops are scalars over possibly-vector SSA defs (e.g. a dvec4
 * load), so each component is a distinct scalar cached independently.
 */
struct unpack_key {
   nir_def *src;
   unsigned swizzle;
};

struct unpacked_pair {
   nir_def *v3;        /* vec3 folded to leading fp32 exp ~0 (value ~[1,2)). */
   nir_def *shift;     /* int: fp64 biased exp - 0x3FF; 0 for zero/subnormal. */
   nir_def *is_inf;    /* bool: true if any upstream operand had exp=0x7FF.   */
   nir_def *inf_sign;  /* i32: sign bit for inf-construction (bit 31 set).    */
};

static uint32_t
unpack_key_hash(const void *key)
{
   const struct unpack_key *k = key;
   uint32_t h = _mesa_hash_pointer(k->src);
   return _mesa_hash_data_with_seed(&k->swizzle, sizeof(k->swizzle), h);
}

static bool
unpack_key_equal(const void *a, const void *b)
{
   const struct unpack_key *ka = a;
   const struct unpack_key *kb = b;
   return ka->src == kb->src && ka->swizzle == kb->swizzle;
}

/* Identify the tf96 softfp64 library by the label stamped on its
 * shader in glsl_float64_funcs_to_nir. Other variants (default fp64,
 * float64q) don't provide the __fp64_unpack_vec3 / __fp64_pack_vec3
 * helpers.
 */
static bool
is_tf96_softfp64(const nir_shader *softfp64)
{
   return softfp64 && softfp64->info.label &&
          strcmp(softfp64->info.label, "float64 tf96") == 0;
}

/* Inline __fp64_unpack_vec3(in uint64_t, out vec3, out int). Void-return
 * GLSL function, so params are [a_in, v3_out, shift_out] in declaration
 * order.
 */
static void
inline_fp64_unpack_v3(nir_builder *b, const nir_shader *softfp64,
                      nir_def *packed,
                      nir_def **out_v3, nir_def **out_shift)
{
   *out_v3 = NULL;
   *out_shift = NULL;
   nir_function *func =
      nir_shader_get_function_for_name(softfp64, "__fp64_unpack_vec3");
   if (!func || !func->impl)
      return;

   nir_variable *a_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "unpack_in");
   nir_deref_instr *a_deref = nir_build_deref_var(b, a_var);
   nir_store_deref(b, a_deref, packed, ~0);

   nir_variable *v3_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "unpack_v3");
   nir_deref_instr *v3_deref = nir_build_deref_var(b, v3_var);

   nir_variable *shift_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "unpack_shift");
   nir_deref_instr *shift_deref = nir_build_deref_var(b, shift_var);

   nir_def *params[3] = {
      &a_deref->def, &v3_deref->def, &shift_deref->def
   };
   nir_inline_function_impl(b, func->impl, params, NULL);

   *out_v3 = nir_load_deref(b, v3_deref);
   *out_shift = nir_load_deref(b, shift_deref);
}

/* Inline __fp64_pack_vec3(in vec3, in int) -> uint64_t. Non-void return:
 * params are [retval_out, v3_in, shift_in].
 */
static nir_def *
inline_fp64_pack_v3(nir_builder *b, const nir_shader *softfp64,
                    nir_def *v3, nir_def *shift)
{
   nir_function *func =
      nir_shader_get_function_for_name(softfp64, "__fp64_pack_vec3");
   if (!func || !func->impl)
      return NULL;

   nir_variable *ret_var =
      nir_local_variable_create(b->impl, glsl_uint64_t_type(), "pack_ret");
   nir_deref_instr *ret_deref = nir_build_deref_var(b, ret_var);

   nir_variable *v3_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "pack_v3");
   nir_deref_instr *v3_deref = nir_build_deref_var(b, v3_var);
   nir_store_deref(b, v3_deref, v3, ~0);

   nir_variable *shift_var =
      nir_local_variable_create(b->impl, glsl_int_type(), "pack_shift");
   nir_deref_instr *shift_deref = nir_build_deref_var(b, shift_var);
   nir_store_deref(b, shift_deref, shift, ~0);

   nir_def *params[3] = {
      &ret_deref->def, &v3_deref->def, &shift_deref->def
   };
   nir_inline_function_impl(b, func->impl, params, NULL);
   return nir_load_deref(b, ret_deref);
}

/* Inline __fmul64_core_unpacked(in vec3, in vec3) -> vec3. Returns the
 * vec3 product (no fold/unfold -- caller tracks shift).
 */
static nir_def *
inline_fmul_core_v3(nir_builder *b, const nir_shader *softfp64,
                    nir_def *a_v3, nir_def *b_v3)
{
   nir_function *func =
      nir_shader_get_function_for_name(softfp64, "__fmul64_core_unpacked");
   if (!func || !func->impl)
      return NULL;

   nir_variable *ret_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fmul_ret");
   nir_deref_instr *ret_deref = nir_build_deref_var(b, ret_var);

   nir_variable *a_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fmul_a");
   nir_deref_instr *a_deref = nir_build_deref_var(b, a_var);
   nir_store_deref(b, a_deref, a_v3, ~0);

   nir_variable *bv_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fmul_b");
   nir_deref_instr *bv_deref = nir_build_deref_var(b, bv_var);
   nir_store_deref(b, bv_deref, b_v3, ~0);

   nir_def *params[3] = {
      &ret_deref->def, &a_deref->def, &bv_deref->def
   };
   nir_inline_function_impl(b, func->impl, params, NULL);
   return nir_load_deref(b, ret_deref);
}

/* Inline __fadd64_core_unpacked(in vec3, in vec3) -> vec3. Caller pre-aligns
 * both operands to a common shift via build_scale_from_delta.
 */
static nir_def *
inline_fadd_core_v3(nir_builder *b, const nir_shader *softfp64,
                    nir_def *a_v3, nir_def *b_v3)
{
   nir_function *func =
      nir_shader_get_function_for_name(softfp64, "__fadd64_core_unpacked");
   if (!func || !func->impl)
      return NULL;

   nir_variable *ret_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fadd_ret");
   nir_deref_instr *ret_deref = nir_build_deref_var(b, ret_var);

   nir_variable *a_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fadd_a");
   nir_deref_instr *a_deref = nir_build_deref_var(b, a_var);
   nir_store_deref(b, a_deref, a_v3, ~0);

   nir_variable *bv_var =
      nir_local_variable_create(b->impl, glsl_vec_type(3), "fadd_b");
   nir_deref_instr *bv_deref = nir_build_deref_var(b, bv_var);
   nir_store_deref(b, bv_deref, b_v3, ~0);

   nir_def *params[3] = {
      &ret_deref->def, &a_deref->def, &bv_deref->def
   };
   nir_inline_function_impl(b, func->impl, params, NULL);
   return nir_load_deref(b, ret_deref);
}

/* Build the fp32 scale factor 2^delta from a signed int delta. Used by
 * the fadd chain to align two (vec3, shift) operands to a common shift:
 * the operand with the smaller shift is scaled down by 2^(shift - smax).
 * For delta below fp32's smallest normal exponent we flush to zero --
 * semantically the same as __fadd64's nExp <= 0 early-out.
 */
static nir_def *
build_scale_from_delta(nir_builder *b, nir_def *delta)
{
   nir_def *bits = nir_ishl_imm(b, nir_iadd_imm(b, delta, 127), 23);
   return nir_bcsel(b, nir_ige_imm(b, delta, -126),
                    bits, nir_imm_float(b, 0.0));
}

/* Get-or-emit the (v3, shift) pair for an fp64 SSA operand. Shared
 * operands across fp64 ops hit the cache and reuse both defs. */
static struct unpacked_pair
get_unpacked_v3(nir_builder *b, struct lower_doubles_data *data,
                nir_alu_src alu_src)
{
   struct unpack_key key = {
      .src = alu_src.src.ssa,
      .swizzle = alu_src.swizzle[0],
   };
   struct hash_entry *e = _mesa_hash_table_search(data->unpacked_of, &key);
   if (e) {
      //fprintf(stderr, "[fp64 cache] UNPACK HIT  src=%p sw=%u\n",
      //        (void *)key.src, key.swizzle);
      return *(struct unpacked_pair *)e->data;
   }
   //fprintf(stderr, "[fp64 cache] UNPACK MISS src=%p sw=%u\n",
   //        (void *)key.src, key.swizzle);

   /* Materialize a scalar uint64 from the (possibly vector + swizzle) src. */
   nir_def *src_scalar = nir_mov_alu(b, alu_src, 1);

   struct unpacked_pair pair;
   inline_fp64_unpack_v3(b, data->softfp64, src_scalar, &pair.v3, &pair.shift);

   /* Derive inf taint from the original packed bits so chain ops can
    * propagate inf correctness through the cache. */
   nir_def *hi = nir_unpack_64_2x32_split_y(b, src_scalar);
   nir_def *biased_exp = nir_iand_imm(b, nir_ushr_imm(b, hi, 20), 0x7FF);
   pair.is_inf = nir_ieq_imm(b, biased_exp, 0x7FF);
   pair.inf_sign = nir_iand_imm(b, hi, 0x80000000);

   if (pair.v3 && pair.shift) {
      struct unpack_key *entry_key =
         ralloc(data->unpacked_of, struct unpack_key);
      *entry_key = key;
      struct unpacked_pair *entry =
         ralloc(data->unpacked_of, struct unpacked_pair);
      *entry = pair;
      _mesa_hash_table_insert(data->unpacked_of, entry_key, entry);
   }
   return pair;
}

/* Insert (packed -> unpacked_pair) into the cache so downstream ops
 * consuming `packed` hit the cache and skip the unpack.
 */
static void
cache_pack_output(struct lower_doubles_data *data, nir_def *packed,
                  struct unpacked_pair pair)
{
   struct unpack_key *entry_key =
      ralloc(data->unpacked_of, struct unpack_key);
   entry_key->src = packed;
   entry_key->swizzle = 0;
   struct unpacked_pair *entry =
      ralloc(data->unpacked_of, struct unpacked_pair);
   *entry = pair;
   _mesa_hash_table_insert(data->unpacked_of, entry_key, entry);
}

/* Lower a 64-bit fmul: unpack both operands, multiply the vec3s, add
 * shifts, pack. Cache the output's (v3, shift, is_inf, inf_sign) so the
 * next op consuming the packed result skips the unpack.
 */
static nir_def *
lower_fmul_chained(nir_builder *b, struct lower_doubles_data *data,
                   nir_alu_instr *instr)
{
   struct unpacked_pair a = get_unpacked_v3(b, data, instr->src[0]);
   struct unpacked_pair bp = get_unpacked_v3(b, data, instr->src[1]);
   if (!a.v3 || !bp.v3)
      return NULL;

   nir_def *rv3 = inline_fmul_core_v3(b, data->softfp64, a.v3, bp.v3);
   if (!rv3)
      return NULL;

   nir_def *rshift = nir_iadd(b, a.shift, bp.shift);

   /* Propagate inf taint through the chain. */
   nir_def *r_is_inf = nir_ior(b, a.is_inf, bp.is_inf);
   nir_def *r_inf_sign = nir_ixor(b, a.inf_sign, bp.inf_sign);

   nir_def *chain_packed =
      inline_fp64_pack_v3(b, data->softfp64, rv3, rshift);
   if (!chain_packed)
      return NULL;
   nir_def *inf_hi = nir_ior_imm(b, r_inf_sign, 0x7FF00000);
   nir_def *inf_result =
      nir_pack_64_2x32_split(b, nir_imm_int(b, 0), inf_hi);
   nir_def *packed = nir_bcsel(b, r_is_inf, inf_result, chain_packed);

   cache_pack_output(data, packed, (struct unpacked_pair) {
      .v3 = rv3, .shift = rshift,
      .is_inf = r_is_inf, .inf_sign = r_inf_sign,
   });

   return packed;
}

/* Lower a 64-bit fadd: unpack both operands, align them to a common
 * shift (max(sa, sb)) by scaling the smaller-shift operand's vec3 down
 * by 2^delta, run __fadd64_unpacked on the aligned vec3s, pack with the
 * shared shift. Propagate inf taint and emit the inf bcsel fallback the
 * same way lower_fmul_chained does.
 */
static nir_def *
lower_fadd_chained(nir_builder *b, struct lower_doubles_data *data,
                   nir_alu_instr *instr)
{
   struct unpacked_pair a = get_unpacked_v3(b, data, instr->src[0]);
   struct unpacked_pair bp = get_unpacked_v3(b, data, instr->src[1]);
   if (!a.v3 || !bp.v3)
      return NULL;

   nir_def *rshift   = nir_imax(b, a.shift, bp.shift);
   nir_def *delta_a  = nir_isub(b, a.shift, rshift);
   nir_def *delta_b  = nir_isub(b, bp.shift, rshift);
   nir_def *scale_a  = build_scale_from_delta(b, delta_a);
   nir_def *scale_b  = build_scale_from_delta(b, delta_b);
   nir_def *a_aln    = nir_fmul(b, a.v3, scale_a);
   nir_def *b_aln    = nir_fmul(b, bp.v3, scale_b);

   nir_def *rv3 = inline_fadd_core_v3(b, data->softfp64, a_aln, b_aln);
   if (!rv3)
      return NULL;

   /* Inf propagation for fadd:
    *   r_is_inf  = a.is_inf || b.is_inf
    *   r_inf_sign = a's sign if a is inf, else b's sign
    * This is correct for single-inf cases and both-inf-same-sign. Both-inf
    * -opposite-sign (should be NaN) falls through as a signed inf -- same
    * approximation as the existing fmul chain for inf*0.
    */
   nir_def *r_is_inf = nir_ior(b, a.is_inf, bp.is_inf);
   nir_def *r_inf_sign = nir_bcsel(b, a.is_inf, a.inf_sign, bp.inf_sign);
   nir_def *chain_packed = inline_fp64_pack_v3(b, data->softfp64, rv3, rshift);

   if (!chain_packed)
      return NULL;

   nir_def *inf_hi = nir_ior_imm(b, r_inf_sign, 0x7FF00000);
   nir_def *inf_result = nir_pack_64_2x32_split(b, nir_imm_int(b, 0), inf_hi);
   nir_def *packed = nir_bcsel(b, r_is_inf, inf_result, chain_packed);

   cache_pack_output(data, packed, (struct unpacked_pair) {
      .v3 = rv3, .shift = rshift,
      .is_inf = r_is_inf, .inf_sign = r_inf_sign,
   });

   return packed;
}


/*
 * Lowers some unsupported double operations, using only:
 *
 * - pack/unpackDouble2x32
 * - conversion to/from single-precision
 * - double add, mul, and fma
 * - conditional select
 * - 32-bit integer and floating point arithmetic
 */

/* Creates a double with the exponent bits set to a given integer value */
static nir_def *
set_exponent(nir_builder *b, nir_def *src, nir_def *exp)
{
   /* Split into bits 0-31 and 32-63 */
   nir_def *lo = nir_unpack_64_2x32_split_x(b, src);
   nir_def *hi = nir_unpack_64_2x32_split_y(b, src);

   /* The exponent is bits 52-62, or 20-30 of the high word, so set the exponent
    * to 1023
    */
   nir_def *new_hi = nir_bitfield_insert(b, hi, exp,
                                         nir_imm_int(b, 20),
                                         nir_imm_int(b, 11));
   /* recombine */
   return nir_pack_64_2x32_split(b, lo, new_hi);
}

static nir_def *
get_exponent(nir_builder *b, nir_def *src)
{
   /* get bits 32-63 */
   nir_def *hi = nir_unpack_64_2x32_split_y(b, src);

   /* extract bits 20-30 of the high word */
   return nir_ubitfield_extract(b, hi, nir_imm_int(b, 20), nir_imm_int(b, 11));
}

/* Return infinity with the sign of the given source which is +/-0 */

static nir_def *
get_signed_inf(nir_builder *b, nir_def *zero)
{
   nir_def *zero_hi = nir_unpack_64_2x32_split_y(b, zero);

   /* The bit pattern for infinity is 0x7ff0000000000000, where the sign bit
    * is the highest bit. Only the sign bit can be non-zero in the passed in
    * source. So we essentially need to OR the infinity and the zero, except
    * the low 32 bits are always 0 so we can construct the correct high 32
    * bits and then pack it together with zero low 32 bits.
    */
   nir_def *inf_hi = nir_ior_imm(b, zero_hi, 0x7ff00000);
   return nir_pack_64_2x32_split(b, nir_imm_int(b, 0), inf_hi);
}

/* Return a correctly signed zero based on src, if we care. */
static nir_def *
get_signed_zero(nir_builder *b, nir_def *src)
{
   uint32_t exec_mode = b->fp_fast_math;

   nir_def *zero;
   if (nir_is_float_control_signed_zero_preserve(exec_mode, 64)) {
      nir_def *hi = nir_unpack_64_2x32_split_y(b, src);
      nir_def *sign = nir_iand_imm(b, hi, 0x80000000);
      zero = nir_pack_64_2x32_split(b, nir_imm_int(b, 0), sign);
   } else {
      zero = nir_imm_double(b, 0.0f);
   }

   return zero;
}

static nir_def *
preserve_nan(nir_builder *b, nir_def *src, nir_def *res)
{
   uint32_t exec_mode = b->fp_fast_math;

   if (nir_is_float_control_nan_preserve(exec_mode, 64)) {
      nir_def *is_nan = nir_fneu(b, src, src);
      return nir_bcsel(b, is_nan, src, res);
   }

   return res;
}

/*
 * Generates the correctly-signed infinity if the source was zero, and flushes
 * the result to 0 if the source was infinity or the calculated exponent was
 * too small to be representable.
 */

static nir_def *
fix_inv_result(nir_builder *b, nir_def *res, nir_def *src,
               nir_def *exp)
{
   /* If the exponent is too small or the original input was infinity,
    * force the result to 0 (flush denorms) to avoid the work of handling
    * denorms properly. If we are asked to preserve NaN, do so, otherwise
    * we return the flushed result for it.
    */
   res = nir_bcsel(b, nir_ior(b, nir_ile_imm(b, exp, 0), nir_feq_imm(b, nir_fabs(b, src), INFINITY)),
                   get_signed_zero(b, src), res);
   res = preserve_nan(b, src, res);

   /* If the original input was 0, generate the correctly-signed infinity */
   res = nir_bcsel(b, nir_fneu_imm(b, src, 0.0f),
                   res, get_signed_inf(b, src));

   return res;
}

static nir_def *
lower_rcp(nir_builder *b, nir_def *src)
{
   /* normalize the input to avoid range issues */
   nir_def *src_norm = set_exponent(b, src, nir_imm_int(b, 1023));

   /* cast to float, do an rcp, and then cast back to get an approximate
    * result
    */
   nir_def *ra = nir_f2f64(b, nir_frcp(b, nir_f2f32(b, src_norm)));

   /* Fixup the exponent of the result - note that we check if this is too
    * small below.
    */
   nir_def *new_exp = nir_isub(b, get_exponent(b, ra),
                               nir_iadd_imm(b, get_exponent(b, src),
                                            -1023));

   ra = set_exponent(b, ra, new_exp);

   /* Do a few Newton-Raphson steps to improve precision.
    *
    * Each step doubles the precision, and we started off with around 24 bits,
    * so we only need to do 2 steps to get to full precision. The step is:
    *
    * x_new = x * (2 - x*src)
    *
    * But we can re-arrange this to improve precision by using another fused
    * multiply-add:
    *
    * x_new = x + x * (1 - x*src)
    *
    * See https://en.wikipedia.org/wiki/Division_algorithm for more details.
    */

   ra = nir_ffma(b, nir_fneg(b, ra), nir_ffma_imm2(b, ra, src, -1), ra);
   ra = nir_ffma(b, nir_fneg(b, ra), nir_ffma_imm2(b, ra, src, -1), ra);

   return fix_inv_result(b, ra, src, new_exp);
}

static nir_def *
lower_sqrt_rsq(nir_builder *b, nir_def *src, bool sqrt)
{
   /* We want to compute:
    *
    * 1/sqrt(m * 2^e)
    *
    * When the exponent is even, this is equivalent to:
    *
    * 1/sqrt(m) * 2^(-e/2)
    *
    * and then the exponent is odd, this is equal to:
    *
    * 1/sqrt(m * 2) * 2^(-(e - 1)/2)
    *
    * where the m * 2 is absorbed into the exponent. So we want the exponent
    * inside the square root to be 1 if e is odd and 0 if e is even, and we
    * want to subtract off e/2 from the final exponent, rounded to negative
    * infinity. We can do the former by first computing the unbiased exponent,
    * and then AND'ing it with 1 to get 0 or 1, and we can do the latter by
    * shifting right by 1.
    */

   nir_def *unbiased_exp = nir_iadd_imm(b, get_exponent(b, src),
                                        -1023);
   nir_def *even = nir_iand_imm(b, unbiased_exp, 1);
   nir_def *half = nir_ishr_imm(b, unbiased_exp, 1);

   nir_def *src_norm = set_exponent(b, src,
                                    nir_iadd_imm(b, even, 1023));

   nir_def *ra = nir_f2f64(b, nir_frsq(b, nir_f2f32(b, src_norm)));
   nir_def *new_exp = nir_isub(b, get_exponent(b, ra), half);
   ra = set_exponent(b, ra, new_exp);

   /*
    * The following implements an iterative algorithm that's very similar
    * between sqrt and rsqrt. We start with an iteration of Goldschmit's
    * algorithm, which looks like:
    *
    * a = the source
    * y_0 = initial (single-precision) rsqrt estimate
    *
    * h_0 = .5 * y_0
    * g_0 = a * y_0
    * r_0 = .5 - h_0 * g_0
    * g_1 = g_0 * r_0 + g_0
    * h_1 = h_0 * r_0 + h_0
    *
    * Now g_1 ~= sqrt(a), and h_1 ~= 1/(2 * sqrt(a)). We could continue
    * applying another round of Goldschmit, but since we would never refer
    * back to a (the original source), we would add too much rounding error.
    * So instead, we do one last round of Newton-Raphson, which has better
    * rounding characteristics, to get the final rounding correct. This is
    * split into two cases:
    *
    * 1. sqrt
    *
    * Normally, doing a round of Newton-Raphson for sqrt involves taking a
    * reciprocal of the original estimate, which is slow since it isn't
    * supported in HW. But we can take advantage of the fact that we already
    * computed a good estimate of 1/(2 * g_1) by rearranging it like so:
    *
    * g_2 = .5 * (g_1 + a / g_1)
    *     = g_1 + .5 * (a / g_1 - g_1)
    *     = g_1 + (.5 / g_1) * (a - g_1^2)
    *     = g_1 + h_1 * (a - g_1^2)
    *
    * The second term represents the error, and by splitting it out we can get
    * better precision by computing it as part of a fused multiply-add. Since
    * both Newton-Raphson and Goldschmit approximately double the precision of
    * the result, these two steps should be enough.
    *
    * 2. rsqrt
    *
    * First off, note that the first round of the Goldschmit algorithm is
    * really just a Newton-Raphson step in disguise:
    *
    * h_1 = h_0 * (.5 - h_0 * g_0) + h_0
    *     = h_0 * (1.5 - h_0 * g_0)
    *     = h_0 * (1.5 - .5 * a * y_0^2)
    *     = (.5 * y_0) * (1.5 - .5 * a * y_0^2)
    *
    * which is the standard formula multiplied by .5. Unlike in the sqrt case,
    * we don't need the inverse to do a Newton-Raphson step; we just need h_1,
    * so we can skip the calculation of g_1. Instead, we simply do another
    * Newton-Raphson step:
    *
    * y_1 = 2 * h_1
    * r_1 = .5 - h_1 * y_1 * a
    * y_2 = y_1 * r_1 + y_1
    *
    * Where the difference from Goldschmit is that we calculate y_1 * a
    * instead of using g_1. Doing it this way should be as fast as computing
    * y_1 up front instead of h_1, and it lets us share the code for the
    * initial Goldschmit step with the sqrt case.
    *
    * Putting it together, the computations are:
    *
    * h_0 = .5 * y_0
    * g_0 = a * y_0
    * r_0 = .5 - h_0 * g_0
    * h_1 = h_0 * r_0 + h_0
    * if sqrt:
    *    g_1 = g_0 * r_0 + g_0
    *    r_1 = a - g_1 * g_1
    *    g_2 = h_1 * r_1 + g_1
    * else:
    *    y_1 = 2 * h_1
    *    r_1 = .5 - y_1 * (h_1 * a)
    *    y_2 = y_1 * r_1 + y_1
    *
    * For more on the ideas behind this, see "Software Division and Square
    * Root Using Goldschmit's Algorithms" by Markstein and the Wikipedia page
    * on square roots
    * (https://en.wikipedia.org/wiki/Methods_of_computing_square_roots).
    */

   nir_def *one_half = nir_imm_double(b, 0.5);
   nir_def *h_0 = nir_fmul(b, one_half, ra);
   nir_def *g_0 = nir_fmul(b, src, ra);
   nir_def *r_0 = nir_ffma(b, nir_fneg(b, h_0), g_0, one_half);
   nir_def *h_1 = nir_ffma(b, h_0, r_0, h_0);
   nir_def *res;
   if (sqrt) {
      nir_def *g_1 = nir_ffma(b, g_0, r_0, g_0);
      nir_def *r_1 = nir_ffma(b, nir_fneg(b, g_1), g_1, src);
      res = nir_ffma(b, h_1, r_1, g_1);
   } else {
      nir_def *y_1 = nir_fmul_imm(b, h_1, 2.0);
      nir_def *r_1 = nir_ffma(b, nir_fneg(b, y_1), nir_fmul(b, h_1, src),
                              one_half);
      res = nir_ffma(b, y_1, r_1, y_1);
   }

   uint32_t exec_mode = b->fp_fast_math;
   if (sqrt) {
      /* Here, the special cases we need to handle are
       * 0 -> 0 (sign preserving)
       * +inf -> +inf
       * -inf -> NaN
       * NaN -> NaN
       */
      /* Denorm flushing/preserving isn't part of the per-instruction bits, so
       * check the execution mode for it.
       */
      uint32_t shader_exec_mode = b->shader->info.float_controls_execution_mode;
      nir_def *src_flushed = src;
      if (!nir_is_denorm_preserve(shader_exec_mode, 64)) {
         src_flushed = nir_bcsel(b,
                                 nir_flt_imm(b, nir_fabs(b, src), DBL_MIN),
                                 get_signed_zero(b, src),
                                 src);
      }
      res = nir_bcsel(b, nir_ior(b, nir_feq_imm(b, src_flushed, 0.0), nir_feq_imm(b, src, INFINITY)),
                      src_flushed, res);
      res = preserve_nan(b, src, res);
   } else {
      res = fix_inv_result(b, res, src, new_exp);
   }

   if (nir_is_float_control_nan_preserve(exec_mode, 64))
      res = nir_bcsel(b, nir_feq_imm(b, src, -INFINITY),
                      nir_imm_double(b, NAN), res);

   return res;
}

static nir_def *
lower_trunc(nir_builder *b, nir_def *src)
{
   nir_def *unbiased_exp = nir_iadd_imm(b, get_exponent(b, src),
                                        -1023);

   nir_def *frac_bits = nir_isub_imm(b, 52, unbiased_exp);

   /*
    * Decide the operation to apply depending on the unbiased exponent:
    *
    * if (unbiased_exp < 0)
    *    return 0
    * else if (unbiased_exp > 52)
    *    return src
    * else
    *    return src & (~0 << frac_bits)
    *
    * Notice that the else branch is a 64-bit integer operation that we need
    * to implement in terms of 32-bit integer arithmetics (at least until we
    * support 64-bit integer arithmetics).
    */

   /* Compute "~0 << frac_bits" in terms of hi/lo 32-bit integer math */
   nir_def *mask_lo =
      nir_bcsel(b,
                nir_ige_imm(b, frac_bits, 32),
                nir_imm_int(b, 0),
                nir_ishl(b, nir_imm_int(b, ~0), frac_bits));

   nir_def *mask_hi =
      nir_bcsel(b,
                nir_ilt_imm(b, frac_bits, 33),
                nir_imm_int(b, ~0),
                nir_ishl(b,
                         nir_imm_int(b, ~0),
                         nir_iadd_imm(b, frac_bits, -32)));

   nir_def *src_lo = nir_unpack_64_2x32_split_x(b, src);
   nir_def *src_hi = nir_unpack_64_2x32_split_y(b, src);

   return nir_bcsel(b,
                    nir_ilt_imm(b, unbiased_exp, 0),
                    get_signed_zero(b, src),
                    nir_bcsel(b, nir_ige_imm(b, unbiased_exp, 53),
                              src,
                              nir_pack_64_2x32_split(b,
                                                     nir_iand(b, mask_lo, src_lo),
                                                     nir_iand(b, mask_hi, src_hi))));
}

static nir_def *
lower_floor(nir_builder *b, nir_def *src)
{
   /*
    * For x >= 0, floor(x) = trunc(x)
    * For x < 0,
    *    - if x is integer, floor(x) = x
    *    - otherwise, floor(x) = trunc(x) - 1
    */
   nir_def *tr = nir_ftrunc(b, src);
   nir_def *positive = nir_fge_imm(b, src, 0.0);
   return nir_bcsel(b,
                    nir_ior(b, positive, nir_feq(b, src, tr)),
                    tr,
                    nir_fadd_imm(b, tr, -1.0));
}

static nir_def *
lower_ceil(nir_builder *b, nir_def *src)
{
   /* if x < 0,                    ceil(x) = trunc(x)
    * else if (x - trunc(x) == 0), ceil(x) = x
    * else,                        ceil(x) = trunc(x) + 1
    */
   nir_def *tr = nir_ftrunc(b, src);
   nir_def *negative = nir_flt_imm(b, src, 0.0);
   return nir_bcsel(b,
                    nir_ior(b, negative, nir_feq(b, src, tr)),
                    tr,
                    nir_fadd_imm(b, tr, 1.0));
}

static nir_def *
lower_fract(nir_builder *b, nir_def *src)
{
   return nir_fsub(b, src, nir_ffloor(b, src));
}

static nir_def *
lower_round_even(nir_builder *b, nir_def *src)
{
   /* Add and subtract 2**52 to round off any fractional bits. */
   nir_def *two52 = nir_imm_double(b, (double)(1ull << 52));
   nir_def *sign = nir_iand_imm(b, nir_unpack_64_2x32_split_y(b, src),
                                1ull << 31);

   b->exact = true;
   nir_def *res = nir_fsub(b, nir_fadd(b, nir_fabs(b, src), two52), two52);
   b->exact = false;

   return nir_bcsel(b, nir_flt(b, nir_fabs(b, src), two52),
                    nir_pack_64_2x32_split(b, nir_unpack_64_2x32_split_x(b, res),
                                           nir_ior(b, nir_unpack_64_2x32_split_y(b, res), sign)),
                    src);
}

static nir_def *
lower_mod(nir_builder *b, nir_def *src0, nir_def *src1)
{
   /* mod(x,y) = x - y * floor(x/y)
    *
    * If the division is lowered, it could add some rounding errors that make
    * floor() to return the quotient minus one when x = N * y. If this is the
    * case, we should return zero because mod(x, y) output value is [0, y).
    * But fortunately Vulkan spec allows this kind of errors; from Vulkan
    * spec, appendix A (Precision and Operation of SPIR-V instructions:
    *
    *   "The OpFRem and OpFMod instructions use cheap approximations of
    *   remainder, and the error can be large due to the discontinuity in
    *   trunc() and floor(). This can produce mathematically unexpected
    *   results in some cases, such as FMod(x,x) computing x rather than 0,
    *   and can also cause the result to have a different sign than the
    *   infinitely precise result."
    *
    * In practice this means the output value is actually in the interval
    * [0, y].
    *
    * While Vulkan states this behaviour explicitly, OpenGL does not, and thus
    * we need to assume that value should be in range [0, y); but on the other
    * hand, mod(a,b) is defined as "a - b * floor(a/b)" and OpenGL allows for
    * some error in division, so a/a could actually end up being 1.0 - 1ULP;
    * so in this case floor(a/a) would end up as 0, and hence mod(a,a) == a.
    *
    * In summary, in the practice mod(a,a) can be "a" both for OpenGL and
    * Vulkan.
    */
   nir_def *floor = nir_ffloor(b, nir_fdiv(b, src0, src1));

   return nir_fsub(b, src0, nir_fmul(b, src1, floor));
}

static nir_def *
lower_minmax(nir_builder *b, nir_op cmp, nir_def *src0, nir_def *src1)
{
   b->exact = true;
   nir_def *src1_is_nan = nir_fneu(b, src1, src1);
   nir_def *cmp_res = nir_build_alu2(b, cmp, src0, src1);
   b->exact = false;
   nir_def *take_src0 = nir_ior(b, src1_is_nan, cmp_res);

   /* IEEE-754-2019 requires that fmin/fmax compare -0 < 0, but -0 and 0 are
    * indistinguishable for flt/fge. So, we fix up signed zeroes.
    */
   if (nir_is_float_control_signed_zero_preserve(b->fp_fast_math, 64)) {
      nir_def *src0_is_negzero = nir_ieq_imm(b, src0, 1ull << 63);
      nir_def *src1_is_poszero = nir_ieq_imm(b, src1, 0x0);
      nir_def *neg_pos_zero = nir_iand(b, src0_is_negzero, src1_is_poszero);

      if (cmp == nir_op_flt) {
         take_src0 = nir_ior(b, take_src0, neg_pos_zero);
      } else {
         assert(cmp == nir_op_fge);
         take_src0 = nir_iand(b, take_src0, nir_inot(b, neg_pos_zero));
      }
   }

   return nir_bcsel(b, take_src0, src0, src1);
}

static nir_def *
lower_sat(nir_builder *b, nir_def *src)
{
   b->exact = true;
   /* This will get lowered again if nir_lower_dminmax is set */
   nir_def *sat = nir_fclamp(b, src, nir_imm_double(b, 0),
                             nir_imm_double(b, 1));
   b->exact = false;
   return sat;
}

static nir_def *
lower_doubles_instr_to_soft(nir_builder *b, nir_alu_instr *instr,
                            const struct lower_doubles_data *data)
{
   nir_lower_doubles_options options = data->options;

   if (!(options & nir_lower_fp64_full_software))
      return NULL;

   /* 64-bit fmul uses the chain-optimized path: operands are unpacked
    * once into (vec3, shift) pairs, cached, and combined at NIR level.
    * The final pack writes the shift back. See struct unpacked_pair.
    */
   if (instr->op == nir_op_fmul && data->unpacked_of) {
      nir_def *r = lower_fmul_chained(b, (struct lower_doubles_data *)data, instr);
      if (r)
         return r;
      /* Fall through to the generic packed path if setup failed. */
   }

   if (instr->op == nir_op_fadd && data->unpacked_of) {
      nir_def *r = lower_fadd_chained(b, (struct lower_doubles_data *)data, instr);
      if (r)
         return r;
      /* Fall through to the generic packed path if setup failed. */
   }

   const char *name;
   const char *mangled_name;
   const struct glsl_type *return_type = glsl_uint64_t_type();
   const nir_shader *softfp64 = data->softfp64;
   bool unpack = false;

   switch (instr->op) {
   case nir_op_f2i64:
      if (instr->src[0].src.ssa->bit_size != 64)
         return false;
      name = "__fp64_to_int64";
      mangled_name = "__fp64_to_int64(u641;";
      return_type = glsl_int64_t_type();
      break;
   case nir_op_f2u64:
      if (instr->src[0].src.ssa->bit_size != 64)
         return false;
      name = "__fp64_to_uint64";
      mangled_name = "__fp64_to_uint64(u641;";
      break;
   case nir_op_f2f64:
      name = "__fp32_to_fp64";
      mangled_name = "__fp32_to_fp64(f1;";
      unpack = true;
      break;
   case nir_op_f2f32:
      name = "__fp64_to_fp32";
      mangled_name = "__fp64_to_fp32(u641;";
      return_type = glsl_float_type();
      unpack = true;
      break;
   case nir_op_f2i32:
      name = "__fp64_to_int";
      mangled_name = "__fp64_to_int(u641;";
      return_type = glsl_int_type();
      break;
   case nir_op_f2u32:
      name = "__fp64_to_uint";
      mangled_name = "__fp64_to_uint(u641;";
      return_type = glsl_uint_type();
      break;
   case nir_op_b2f64:
      name = "__bool_to_fp64";
      mangled_name = "__bool_to_fp64(b1;";
      break;
   case nir_op_i2f64:
      if (instr->src[0].src.ssa->bit_size == 64) {
         name = "__int64_to_fp64";
         mangled_name = "__int64_to_fp64(i641;";
      } else {
         name = "__int_to_fp64";
         mangled_name = "__int_to_fp64(i1;";
      }
      break;
   case nir_op_u2f64:
      if (instr->src[0].src.ssa->bit_size == 64) {
         name = "__uint64_to_fp64";
         mangled_name = "__uint64_to_fp64(u641;";
      } else {
         name = "__uint_to_fp64";
         mangled_name = "__uint_to_fp64(u1;";
      }
      break;
   case nir_op_fabs:
      name = "__fabs64";
      mangled_name = "__fabs64(u641;";
      break;
   case nir_op_fneg:
      name = "__fneg64";
      mangled_name = "__fneg64(u641;";
      unpack = true;
      break;
   case nir_op_fround_even:
      name = "__fround64";
      mangled_name = "__fround64(u641;";
      break;
   case nir_op_ftrunc:
      name = "__ftrunc64";
      mangled_name = "__ftrunc64(u641;";
      break;
   case nir_op_ffloor:
      name = "__ffloor64";
      mangled_name = "__ffloor64(u641;";
      break;
   case nir_op_ffract:
      name = "__ffract64";
      mangled_name = "__ffract64(u641;";
      break;
   case nir_op_fsign:
      name = "__fsign64";
      mangled_name = "__fsign64(u641;";
      break;
   case nir_op_feq:
      name = "__feq64";
      mangled_name = "__feq64(u641;u641;";
      return_type = glsl_bool_type();
      break;
   case nir_op_fneu:
      name = "__fneu64";
      mangled_name = "__fneu64(u641;u641;";
      return_type = glsl_bool_type();
      unpack = true;
      break;
   case nir_op_flt:
      name = "__flt64";
      mangled_name = "__flt64(u641;u641;";
      return_type = glsl_bool_type();
      unpack = true;
      break;
   case nir_op_fge:
      name = "__fge64";
      mangled_name = "__fge64(u641;u641;";
      return_type = glsl_bool_type();
      unpack = true;
      break;
   case nir_op_fmin:
      name = "__fmin64";
      mangled_name = "__fmin64(u641;u641;";
      unpack = true;
      break;
   case nir_op_fmax:
      name = "__fmax64";
      mangled_name = "__fmax64(u641;u641;";
      unpack = true;
      break;
   case nir_op_fadd:
      name = "__fadd64";
      mangled_name = "__fadd64(u641;u641;";
      unpack = true;
      break;
   case nir_op_fmul:
      name = "__fmul64";
      mangled_name = "__fmul64(u641;u641;";
      unpack = true;
      break;
   case nir_op_ffma:
      name = "__ffma64";
      mangled_name = "__ffma64(u641;u641;u641;";
      unpack = true;
      break;
   case nir_op_fsat:
      name = "__fsat64";
      mangled_name = "__fsat64(u641;";
      break;
   case nir_op_fisfinite:
      name = "__fisfinite64";
      mangled_name = "__fisfinite64(u641;";
      return_type = glsl_bool_type();
      break;
   case nir_op_fsqrt:
      name = "__fsqrt64";
      mangled_name = "__fsqrt64(u641;";
      unpack = true;
      break;
   default:
      if (is_quick_softfp64(softfp64)) {
         switch (instr->op) {
            case nir_op_frcp:
               name = "__frcp64";
               mangled_name = "__frcp64(u641;";
               unpack = true;
               break;
            case nir_op_frsq:
               name = "__frsq64";
               mangled_name = "__frsq64(u641;";
               unpack = true;
               break;
            case nir_op_fdiv:
               name = "__fdiv64";
               mangled_name = "__fdiv64(u641;u641;";
               unpack = true;
               break;
            case nir_op_fsub:
               name = "__fsub64";
               mangled_name = "__fsub64(u641;u641;";
               unpack = true;
               break;
            default:
               return pack_double_before_lower_alu(b, instr, data);
         }
      } else {
         return false;
      }
   }

   assert(softfp64 != NULL);
   nir_function *func = nir_shader_get_function_for_name(softfp64, name);

   /* Another attempt, but this time with mangled names if softfp64
    * shader is taken from SPIR-V.
    */
   if (!func)
      func = nir_shader_get_function_for_name(softfp64, mangled_name);

   if (!func || !func->impl) {
      fprintf(stderr, "Cannot find function \"%s\"\n", name);
      assert(func);
   }

   nir_def *params[4] = {
      NULL,
   };

   nir_variable *ret_tmp =
      nir_local_variable_create(b->impl, return_type, "return_tmp");
   nir_deref_instr *ret_deref = nir_build_deref_var(b, ret_tmp);
   params[0] = &ret_deref->def;

   assert(nir_op_infos[instr->op].num_inputs + 1 == func->num_params);
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      nir_alu_type n_type =
         nir_alu_type_get_base_type(nir_op_infos[instr->op].input_types[i]);
      /* Add bitsize */
      n_type = n_type | instr->src[0].src.ssa->bit_size;

      const struct glsl_type *param_type =
         glsl_scalar_type(nir_get_glsl_base_type_for_nir_type(n_type));

      // convert `instr->src[i]` before inline
      nir_def* src_def = NULL;
      if (is_quick_softfp64(softfp64)) {
         src_def = convert_double_before_inline(b, instr->src[i], data, unpack);
      }

      nir_variable *param =
         nir_local_variable_create(b->impl, param_type, "param");
      nir_deref_instr *param_deref = nir_build_deref_var(b, param);
      nir_store_deref(b, param_deref, src_def ? src_def : nir_mov_alu(b, instr->src[i], 1), ~0);

      assert(i + 1 < ARRAY_SIZE(params));
      params[i + 1] = &param_deref->def;
   }

   nir_inline_function_impl(b, func->impl, params, NULL);

   if (is_quick_softfp64(softfp64)) {
      nir_def* out_def = nir_load_deref(b, ret_deref);
      if (unpack) {
         struct nir_lower_double_def *key = ralloc(b->shader, struct nir_lower_double_def);
         key->index = 0;
         key->src = out_def;
         key->dest = NULL;
         key->packed = false;
         add_lower_double_def(data->defs, key);
      }
      return out_def;
   } else {
      return nir_load_deref(b, ret_deref);
   }
}

nir_lower_doubles_options
nir_lower_doubles_op_to_options_mask(nir_op opcode)
{
   switch (opcode) {
   case nir_op_frcp:
      return nir_lower_drcp;
   case nir_op_fsqrt:
      return nir_lower_dsqrt;
   case nir_op_frsq:
      return nir_lower_drsq;
   case nir_op_ftrunc:
      return nir_lower_dtrunc;
   case nir_op_ffloor:
      return nir_lower_dfloor;
   case nir_op_fceil:
      return nir_lower_dceil;
   case nir_op_ffract:
      return nir_lower_dfract;
   case nir_op_fround_even:
      return nir_lower_dround_even;
   case nir_op_fmod:
      return nir_lower_dmod;
   case nir_op_fsub:
      return nir_lower_dsub;
   case nir_op_fdiv:
      return nir_lower_ddiv;
   case nir_op_fmin:
   case nir_op_fmax:
      return nir_lower_dminmax;
   case nir_op_fsat:
      return nir_lower_dsat;
   default:
      return 0;
   }
}

static bool
should_lower_double_instr(const nir_instr *instr, const void *_data)
{
   struct lower_doubles_data *data = (struct lower_doubles_data *)_data;
   const nir_lower_doubles_options options = data->options;

   if (instr->type != nir_instr_type_alu) {
      if (data != NULL && is_quick_softfp64(data->softfp64)) {
         return should_lower_double_phi_instr(instr, options);
      } else {
         return false;
      }
   }

   const nir_alu_instr *alu = nir_instr_as_alu(instr);

   bool is_64 = alu->def.bit_size == 64;

   unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
   for (unsigned i = 0; i < num_srcs; i++) {
      is_64 |= (nir_src_bit_size(alu->src[i].src) == 64);
   }

   if (!is_64)
      return false;

   if (options & nir_lower_fp64_full_software)
      return true;

   return options & nir_lower_doubles_op_to_options_mask(alu->op);
}

static nir_def *
lower_doubles_instr(nir_builder *b, nir_instr *instr, void *_data)
{
   const struct lower_doubles_data *data = _data;
   const nir_lower_doubles_options options = data->options;

   if (instr->type == nir_instr_type_phi && data != NULL && is_quick_softfp64(data->softfp64)) {
      return pack_double_before_lower_phi(b, nir_instr_as_phi(instr), data);
   }

   nir_alu_instr *alu = nir_instr_as_alu(instr);

   /* Easier to set it here than pass it around all over ther place. */
   b->fp_fast_math = alu->fp_fast_math;

   nir_def *soft_def = lower_doubles_instr_to_soft(b, alu, data);
   if (soft_def)
      return soft_def;

   if (!(options & nir_lower_doubles_op_to_options_mask(alu->op)))
      return NULL;

   nir_def *src = nir_mov_alu(b, alu->src[0],
                              alu->def.num_components);

   switch (alu->op) {
   case nir_op_frcp:
      return lower_rcp(b, src);
   case nir_op_fsqrt:
      return lower_sqrt_rsq(b, src, true);
   case nir_op_frsq:
      return lower_sqrt_rsq(b, src, false);
   case nir_op_ftrunc:
      return lower_trunc(b, src);
   case nir_op_ffloor:
      return lower_floor(b, src);
   case nir_op_fceil:
      return lower_ceil(b, src);
   case nir_op_ffract:
      return lower_fract(b, src);
   case nir_op_fround_even:
      return lower_round_even(b, src);
   case nir_op_fsat:
      return lower_sat(b, src);

   case nir_op_fdiv:
   case nir_op_fsub:
   case nir_op_fmod:
   case nir_op_fmin:
   case nir_op_fmax: {
      nir_def *src1 = nir_mov_alu(b, alu->src[1],
                                  alu->def.num_components);
      switch (alu->op) {
      case nir_op_fdiv:
         return nir_fmul(b, src, nir_frcp(b, src1));
      case nir_op_fsub:
         return nir_fadd(b, src, nir_fneg(b, src1));
      case nir_op_fmod:
         return lower_mod(b, src, src1);
      case nir_op_fmin:
         return lower_minmax(b, nir_op_flt, src, src1);
      case nir_op_fmax:
         return lower_minmax(b, nir_op_fge, src, src1);
      default:
         unreachable("unhandled opcode");
      }
   }
   default:
      unreachable("unhandled opcode");
   }
}

/* Chain cache gate: full-software fp64, tf96 library selected, and
 * TF96_CACHE=1 in the environment. Default is off so we don't
 * perturb other softfp64 methods or correctness testing implicitly.
 */
static bool
chain_cache_enabled(const nir_shader *softfp64,
                    nir_lower_doubles_options options)
{
   if (!(options & nir_lower_fp64_full_software))
      return false;
   if (!is_tf96_softfp64(softfp64))
      return false;
   const char *env = getenv("TF96_CACHE");
   return env && strcmp(env, "1") == 0;
}

static bool
nir_lower_doubles_impl(nir_function_impl *impl,
                       const nir_shader *softfp64,
                       nir_lower_doubles_options options)
{
   struct lower_doubles_data data = {
      .softfp64 = softfp64,
      .options = options,
      .defs = _mesa_set_create(NULL, hash_nir_lower_double_def, nir_lower_double_def_equal),
      .unpacked_of = chain_cache_enabled(softfp64, options)
                     ? _mesa_hash_table_create(NULL, unpack_key_hash,
                                               unpack_key_equal)
                     : NULL,
   };

   bool progress =
      nir_function_impl_lower_instructions(impl,
                                           should_lower_double_instr,
                                           lower_doubles_instr,
                                           &data);

   if (data.unpacked_of)
      _mesa_hash_table_destroy(data.unpacked_of, NULL);

   if (progress && (options & nir_lower_fp64_full_software)) {
      /* Indices are completely messed up now */
      nir_index_ssa_defs(impl);

      nir_metadata_preserve(impl, nir_metadata_none);

      /* And we have deref casts we need to clean up thanks to function
       * inlining.
       */
      nir_opt_deref_impl(impl);
   } else if (progress) {
      nir_metadata_preserve(impl, nir_metadata_control_flow);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   if (data.defs) {
      _mesa_set_destroy(data.defs, destroy_lower_double_def);
      data.defs = NULL;
   }
   return progress;
}

bool
nir_lower_doubles(nir_shader *shader,
                  const nir_shader *softfp64,
                  nir_lower_doubles_options options)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      progress |= nir_lower_doubles_impl(impl, softfp64, options);
   }

   return progress;
}
