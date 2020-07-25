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

#include "serene/compiler.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/namespace.hpp"
#include "serene/reader.hpp"
#include "serene/state.hpp"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {

Compiler::Compiler() {
  string default_ns_name("user");
  Namespace *default_ns = new Namespace(default_ns_name);

  builder = new IRBuilder(this->context);
  state = new State();

  state->add_namespace(default_ns, true, true);
};

Value *Compiler::log_error(const char *s) {
  fmt::print("[Error]: {}\n", s);
  return nullptr;
};

void Compiler::compile(string &input) {
  Reader *r = new Reader(input);
  ast_tree &ast = r->read();

  COMPILER_LOG("Parsing the input has been done.")
  for (const ast_node &x : ast) {
    auto *IR{x->codegen(*this, *this->state)};

    if (IR) {
      fmt::print("'{}' generates: \n", x->string_repr()

      );
      IR->print(errs());
      fmt::print("\n");
    } else {
      fmt::print("No gen\n");
    }
  }
  delete r;
  COMPILER_LOG("Done!")
  return;
};

Compiler::~Compiler() {
  COMPILER_LOG("Deleting state...");
  delete state;
  COMPILER_LOG("Deleting builder...");
  delete builder;
}

} // namespace serene
