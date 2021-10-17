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

#ifndef SERENE_EXPRS_TRAITS_H
#define SERENE_EXPRS_TRAITS_H

#include "serene/context.h"
#include "serene/reader/location.h"
#include "serene/reader/traits.h"
#include "serene/traits.h"
#include "serene/utils.h"

namespace serene::exprs {
/// This enum represent the expression type and **not** the value type.
enum class ExprType {
  Symbol,
  List,
  Number,
  Def,
  Error,
  Fn,
  Call,
};

/// The string represantion of built in expr types (NOT DATATYPES).
static const char *exprTypes[] = {
    "Symbol", "List", "Number", "Def", "Error", "Fn", "Call",
};

}; // namespace serene::exprs

#endif
