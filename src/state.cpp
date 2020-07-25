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

#include "serene/state.hpp"
#include "serene/llvm/IR/Value.h"
#include "serene/namespace.hpp"
#include <fmt/core.h>
#include <string>

using namespace std;
using namespace llvm;

namespace serene {
State::State() { current_ns = nullptr; };

void State::add_namespace(Namespace *ns, bool set_current, bool overwrite) {
  if (ns->name.empty()) {
    // TODO: Better error handling
    fmt::print("Error: namespace name is missing\n.");
    exit(1);
  }

  Namespace *already_exist_ns = namespaces[ns->name];

  if (already_exist_ns && !overwrite) {
    return;
  }

  if (already_exist_ns) {
    delete namespaces[ns->name];
  }

  namespaces[ns->name] = ns;

  if (set_current) {
    set_current_ns(ns);
  }
};

bool State::set_current_ns(Namespace *ns) {
  Namespace *already_exist_ns = namespaces[ns->name];
  if (already_exist_ns) {
    current_ns = ns;
    return true;
  }
  return false;
};

Value *State::lookup_in_current_scope(string &name) {
  if (this->current_ns) {
    return current_ns->lookup(name);
  }

  fmt::print("FATAL ERROR: Current ns is not set.");
  // TODO: Come up with the ERRNO table and return the proper ERRNO
  exit(1);
};

State::~State() {
  STATE_LOG("Deleting namespaces...")
  std::map<string, Namespace *>::iterator it = namespaces.begin();
  while (it != namespaces.end()) {
    STATE_LOG("DELETING {}", it->first);
    Namespace *tmp = it->second;
    namespaces[it->first] = nullptr;
    delete tmp;
    it++;
  }
  STATE_LOG("Clearing namespaces...");
  namespaces.clear();
};
} // namespace serene
