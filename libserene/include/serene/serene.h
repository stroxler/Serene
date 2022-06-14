/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef SERENE_SERENE_H
#define SERENE_SERENE_H

#include "serene/export.h"     // for SERENE_EXPORT
#include "serene/jit/halley.h" // for Engine, MaybeEngine
#include "serene/options.h"    // for Options

namespace serene {

/// Clinet applications have to call this function before any interaction
/// with the Serene's compiler API.
void SERENE_EXPORT initSerene();

/// Register the global CLI options of the serene compiler. If the client
/// application needs to setup the compilers options automatically use this
/// function in conjunction with `applySereneCLOptions`.
void SERENE_EXPORT registerSereneCLOptions();

/// Applies the global compiler options on the give \p SereneContext. This
/// function has to be called after `llvm::cl::ParseCommandLineOptions`.
void SERENE_EXPORT applySereneCLOptions(serene::jit::Engine &engine);

serene::jit::MaybeEngine SERENE_EXPORT makeEngine(Options opts = Options());

} // namespace serene
#endif
