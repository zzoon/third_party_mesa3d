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

#include "util/u_nir_opt_cache.h"

#include "util/blob.h"
#include "util/hash_table.h"
#include "util/sha1/sha1.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/nir/nir.h"

#include <string.h>
#include <stdlib.h>

struct nir_opt_cache_entry {
   unsigned char sha1[SHA1_DIGEST_LENGTH];   /* key - hash table key points here */
   size_t size;              /* size of serialized blob */
   uint8_t data[];           /* serialized post-opt NIR (flexible array) */
};

static uint32_t
entry_key_hash(const void *key)
{
   /* Use the first dword of SHA1 as the hash. */
   return *(const uint32_t *)key;
}

static bool
entry_key_equals(const void *a, const void *b)
{
   return memcmp(a, b, SHA1_DIGEST_LENGTH) == 0;
}

void
util_nir_opt_cache_init(struct util_nir_opt_cache *cache)
{
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->table = _mesa_hash_table_create(NULL, entry_key_hash, entry_key_equals);
   cache->hits = 0;
   cache->misses = 0;
}

static void
destroy_entry(struct hash_entry *entry)
{
   free(entry->data);
}

void
util_nir_opt_cache_deinit(struct util_nir_opt_cache *cache)
{
   if (cache->table) {
      _mesa_hash_table_destroy(cache->table, destroy_entry);
      cache->table = NULL;
      simple_mtx_destroy(&cache->lock);
   }
}

struct nir_shader *
util_nir_opt_cache_lookup(struct util_nir_opt_cache *cache,
                          const unsigned char sha1[SHA1_DIGEST_LENGTH],
                          const struct nir_shader_compiler_options *options)
{
   simple_mtx_lock(&cache->lock);
   struct hash_entry *entry = _mesa_hash_table_search(cache->table, sha1);
   struct nir_opt_cache_entry *e =
      entry ? (struct nir_opt_cache_entry *)entry->data : NULL;
   simple_mtx_unlock(&cache->lock);

   if (!e) {
      cache->misses++;
      return NULL;
   }

   cache->hits++;
   struct blob_reader reader;
   blob_reader_init(&reader, e->data, e->size);
   return nir_deserialize(NULL, options, &reader);
}

void
util_nir_opt_cache_insert(struct util_nir_opt_cache *cache,
                          const unsigned char sha1[SHA1_DIGEST_LENGTH],
                          const struct nir_shader *nir)
{
   struct blob blob;
   blob_init(&blob);
   /* strip=false: variable names must be preserved for later uniform lookup. */
   nir_serialize(&blob, nir, false);

   size_t data_size = blob.size;
   struct nir_opt_cache_entry *e = malloc(sizeof(*e) + data_size);
   if (!e) {
      blob_finish(&blob);
      return;
   }
   memcpy(e->sha1, sha1, SHA1_DIGEST_LENGTH);
   e->size = data_size;
   memcpy(e->data, blob.data, data_size);
   blob_finish(&blob);

   simple_mtx_lock(&cache->lock);
   /* Don't insert if already present (concurrent MISS -> insert race). */
   if (!_mesa_hash_table_search(cache->table, e->sha1))
      _mesa_hash_table_insert(cache->table, e->sha1, e);
   else
      free(e);
   simple_mtx_unlock(&cache->lock);
}
