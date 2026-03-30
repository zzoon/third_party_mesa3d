/*
 * Copyright 2026 Igalia S.L.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* Screen-level cache for NIR optimization results.
 *
 * Maps SHA1(pre-opt NIR) → serialized post-opt NIR blob, allowing expensive
 * lowering passes (e.g. nir_lower_doubles for SoftFP64) to be skipped when
 * the same NIR is compiled again.
 *
 */

#ifndef U_NIR_OPT_CACHE_H
#define U_NIR_OPT_CACHE_H

#include "util/simple_mtx.h"
#include "util/sha1/sha1.h"

#ifdef __cplusplus
extern "C" {
#endif

struct hash_table;
struct nir_shader;
struct nir_shader_compiler_options;

struct util_nir_opt_cache {
   simple_mtx_t lock;
   struct hash_table *table;
   unsigned hits, misses;
};

void util_nir_opt_cache_init(struct util_nir_opt_cache *cache);
void util_nir_opt_cache_deinit(struct util_nir_opt_cache *cache);

/**
 * Look up the post-opt NIR for a given pre-opt NIR SHA1.
 * Returns a freshly deserialized nir_shader (allocated as its own ralloc
 * root) on hit, or NULL on miss.
 */
struct nir_shader *
util_nir_opt_cache_lookup(struct util_nir_opt_cache *cache,
                          const unsigned char sha1[SHA1_DIGEST_LENGTH],
                          const struct nir_shader_compiler_options *options);

/**
 * Store the post-opt NIR (serialized) keyed by the pre-opt SHA1.
 * No-op if the key is already present (handles concurrent inserts).
 */
void
util_nir_opt_cache_insert(struct util_nir_opt_cache *cache,
                          const unsigned char sha1[SHA1_DIGEST_LENGTH],
                          const struct nir_shader *nir);

#ifdef __cplusplus
}
#endif

#endif /* U_NIR_OPT_CACHE_H */
