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

#ifndef SERENE_FS_H
#define SERENE_FS_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

#include <filesystem>

#define MAX_PATH_SLOTS 256

namespace serene {
class SereneContext;
}; // namespace serene

namespace serene::fs {

enum class NSFileType {
  Source = 0,
  TextIR,
  BinaryIR,
  ObjectFile,
  StaticLib,
  SharedLib
};

std::string extensionFor(SereneContext &ctx, NSFileType t);
/// Converts the given namespace name `nsName` to the file name
/// for that name space. E.g, `some.random.ns` will be translated
/// to `some_random_ns`.
std::string namespaceToPath(const llvm::StringRef nsName);

/// Return a boolean indicating whether or not the given path exists.
bool exists(llvm::StringRef path);
/// Join the given `path1` and `path2` with respect to the platform
/// conventions.
std::string join(llvm::StringRef path1, llvm::StringRef path2);

}; // namespace serene::fs
#endif
