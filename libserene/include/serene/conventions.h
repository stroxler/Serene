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

#ifndef SERENE_CONVENTIONS_H
#define SERENE_CONVENTIONS_H

#include "serene/config.h"

#include <llvm/ADT/StringRef.h>

#include <string>

namespace serene {
static std::string mangleInternalStringName(llvm::StringRef str) {
  return "__serene__internal__str__" + str.str();
}

static std::string mangleInternalSymName(llvm::StringRef str) {
  return "__serene__symbol__" + str.str();
}

} // namespace serene

#endif
