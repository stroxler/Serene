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

#ifndef SERENE_OPTIONS_H
#define SERENE_OPTIONS_H

#include "serene/export.h"

namespace serene {
/// Options describes the compiler options that can be passed to the
/// compiler via command line. Anything that user should be able to
/// tweak about the compiler has to end up here regardless of the
/// different subsystem that might use it.
struct SERENE_EXPORT Options {

  /// Whether to use colors for the output or not
  bool withColors = true;

  // JIT related flags
  bool JITenableObjectCache              = true;
  bool JITenableGDBNotificationListener  = true;
  bool JITenablePerfNotificationListener = true;
  bool JITLazy                           = false;

  // namespace serene Options() = default;
};
} // namespace serene

#endif
