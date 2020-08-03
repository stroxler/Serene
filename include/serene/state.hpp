/**
 * Serene programming language.
 *
 *  Copyright (c) 2020 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef STATE_H
#define STATE_H

#include "serene/llvm/IR/Value.h"
#include "serene/logger.hpp"
#include "serene/namespace.hpp"
#include <llvm/IR/Module.h>
#include <string>

#if defined(ENABLE_LOG) || defined(ENABLE_STATE_LOG)
#define STATE_LOG(...) __LOG("STATE", __VA_ARGS__);
#else
#define STATE_LOG(...) ;
#endif

namespace serene {
class State {
public:
  std::map<std::string, Namespace *> namespaces;
  Namespace *current_ns;

  State();

  void add_namespace(Namespace *ns, bool set_current, bool overwrite);
  bool set_current_ns(Namespace *ns);
  llvm::Value *lookup_in_current_scope(const std::string &name);
  void set_in_current_ns_root_scope(std::string name, llvm::Value *v);
  ~State();
};
} // namespace serene

#endif
