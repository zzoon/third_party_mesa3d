/*
 * Copyright (c) 2025 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIR_DOUBLE_PACK_RECORDS_H
#define NIR_DOUBLE_PACK_RECORDS_H

#include "util/detect_os.h"
#include "nir.h"
#include "nir_builder.h"

struct nir_lower_double_def {
   nir_def* src;
   nir_def* dest;
   nir_def* shift_dest;   /* tf96 only -- shift SSA from __unpackFp64ToFp32. NULL otherwise. */
   uint32_t index;
   bool packed;
};
struct lower_doubles_data {
   const nir_shader *softfp64;
   nir_lower_doubles_options options;
   struct set *defs;
   /* Transient flag for lazy-pack: set in should_lower_double_instr (the
    * filter) when the alu's uses are all chain-lowerable.
    */
   bool current_lazy_pack;
};
uint32_t hash_nir_lower_double_def(const void *p);
bool nir_lower_double_def_equal(const void *void_a, const void *void_b);
void destroy_lower_double_def(struct set_entry *entry);
void add_lower_double_def(struct set* defs, struct nir_lower_double_def* key);
nir_def *convert_double_before_inline(nir_builder *b, nir_alu_src src, const struct lower_doubles_data *data,
   bool unpack);

/* tf96 helper: retrieve the cached shift SSA produced by an
 * earlier convert_double_before_inline(unpack=true) call.
 */
nir_def *get_tf96_shift_dest(const struct lower_doubles_data *data, nir_alu_src src);

/* tf96 pack: emit __packFp32ToFp64(vec3, int) -> uint64 inline. */
nir_def *convert_double_pack_tf96(nir_builder *b,
                                  nir_def *v3, nir_def *shift,
                                  const struct lower_doubles_data *data);

nir_def *pack_double_before_lower_phi(nir_builder *b, nir_phi_instr *instr, const struct lower_doubles_data *data);
nir_def *pack_double_before_lower_alu(nir_builder *b, nir_alu_instr *instr, const struct lower_doubles_data *data);
bool should_lower_double_phi_instr(const nir_instr *instr, const nir_lower_doubles_options options);
bool is_quick_softfp64(const nir_shader *softfp64);
bool is_tf96_softfp64(const nir_shader *softfp64);
#endif // NIR_DOUBLE_PACK_RECORDS_H
