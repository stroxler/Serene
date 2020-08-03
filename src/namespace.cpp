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

#include "serene/expr.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/special_forms/def.hpp"
#include <fmt/core.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

Namespace::BuiltinMap Namespace::builtins = [] {
  NAMESPACE_LOG("Initializing builtins map.");
  BuiltinMap exprs_map;
  // exprs_map.insert(std::make_pair("def", &special_forms::Def::make));
  // MakerFn def = ;
  exprs_map["def"] = special_forms::Def::make;
  return exprs_map;
}();

Value *Namespace::lookup(const string &name) { return scope[name]; };

void Namespace::insert_symbol(const string &name, Value *v) { scope[name] = v; }

void Namespace::print_scope() {
  typedef map<string, Value *>::const_iterator Iter;

  fmt::print("Scope of '{}' ns.\n", name);
  for (Iter iter = scope.begin(); iter != scope.end(); iter++) {
    fmt::print("{}\n", iter->first);
  }
};

Namespace::~Namespace() {}

} // namespace serene
