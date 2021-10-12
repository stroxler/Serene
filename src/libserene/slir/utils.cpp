/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

#include "serene/context.h"
#include "serene/namespace.h"
#include "serene/reader/location.h"

#include <mlir/IR/BuiltinOps.h>

namespace serene::slir {
::mlir::Location toMLIRLocation(serene::Namespace &ns,
                                serene::reader::Location &loc) {
  mlir::OpBuilder builder(&ns.getContext().mlirContext);
  auto file = ns.filename;
  std::string filename{file.getValueOr("REPL")};

  return mlir::FileLineColLoc::get(builder.getIdentifier(filename), loc.line,
                                   loc.col);
}

} // namespace serene::slir
