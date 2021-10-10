/*
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef SERENE_SERENE_H
#define SERENE_SERENE_H

#include "serene/config.h"
#include "serene/context.h"
#include "serene/export.h"
#include "serene/source_mgr.h"

namespace serene {

/// Clinet applications have to call this function before any interaction with
/// the Serene's compiler API.
SERENE_EXPORT void initCompiler();

/// Register the global CLI options of the serene compiler. If the client
/// application needs to setup the compilers options automatically use this
/// function in conjunction with `applySereneCLOptions`.
SERENE_EXPORT void registerSereneCLOptions();

/// Applies the global compiler options on the give \p SereneContext. This
/// function has to be called after `llvm::cl::ParseCommandLineOptions`.
SERENE_EXPORT void applySereneCLOptions(SereneContext &ctx);
} // namespace serene
#endif
