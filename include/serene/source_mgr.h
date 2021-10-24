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

#ifndef SERENE_SOURCE_MGR_H
#define SERENE_SOURCE_MGR_H

#include "serene/namespace.h"
#include "serene/reader/location.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/Timing.h>

#include <memory>

#define SMGR_LOG(...)                       \
  DEBUG_WITH_TYPE("sourcemgr", llvm::dbgs() \
                                   << "[SMGR]: " << __VA_ARGS__ << "\n");

namespace serene {
class SereneContext;

/// This class is quite similar to the `llvm::SourceMgr` in functionality. We
/// even borrowed some of the code from the original implementation but removed
/// a lot of code that ar irrelevant to us.
///
/// SouceMgr is responsible for finding a namespace in the `loadPaths` and read
/// the content of the `.srn` (or any of the `DEFAULT_SUFFIX`) into a
/// `llvm::MemoryBuffer` embedded in a `SrcBuffer` object as the owner of the
/// source files and then it will call the `reader` on the buffer to parse it
/// and create the actual `Namespace` object from the parsed AST.
///
/// Later on, whenever we need to refer to the source file of a namespace for
/// diagnosis purposes or any other purpose we can use the functions in this
/// class to get hold of a pointer to a specific `reader::Location` of the
/// buffer.
///
/// Note: Unlike the original version, SourceMgr does not handle the diagnostics
/// and it uses the Serene's `DiagnosticEngine` for that matter.
class SERENE_EXPORT SourceMgr {

public:
  // TODO: Make it a vector of supported suffixes
  std::string DEFAULT_SUFFIX = "srn";

private:
  struct SrcBuffer {
    /// The memory buffer for the file.
    std::unique_ptr<llvm::MemoryBuffer> buffer;

    /// Vector of offsets into Buffer at which there are line-endings
    /// (lazily populated). Once populated, the '\n' that marks the end of
    /// line number N from [1..] is at Buffer[OffsetCache[N-1]]. Since
    /// these offsets are in sorted (ascending) order, they can be
    /// binary-searched for the first one after any given offset (eg. an
    /// offset corresponding to a particular SMLoc).
    ///
    /// Since we're storing offsets into relatively small files (often smaller
    /// than 2^8 or 2^16 bytes), we select the offset vector element type
    /// dynamically based on the size of Buffer.
    mutable void *offsetCache = nullptr;

    /// Look up a given \p Ptr in in the buffer, determining which line it came
    /// from.
    unsigned getLineNumber(const char *ptr) const;
    template <typename T>
    unsigned getLineNumberSpecialized(const char *ptr) const;

    /// Return a pointer to the first character of the specified line number or
    /// null if the line number is invalid.
    const char *getPointerForLineNumber(unsigned lineNo) const;

    template <typename T>
    const char *getPointerForLineNumberSpecialized(unsigned lineNo) const;

    /// This is the location of the parent import or unknown location if it is
    /// the main namespace
    reader::LocationRange importLoc;

    SrcBuffer() = default;
    SrcBuffer(SrcBuffer &&) noexcept;
    SrcBuffer(const SrcBuffer &) = delete;
    SrcBuffer &operator=(const SrcBuffer &) = delete;
    ~SrcBuffer();
  };
  using ErrorOrMemBufPtr = llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>;

  /// This is all of the buffers that we are reading from.
  std::vector<SrcBuffer> buffers;

  /// A hashtable that works as an index from namespace names to the buffer
  /// position it the `buffer`
  llvm::StringMap<unsigned> nsTable;

  // This is the list of directories we should search for include files in.
  std::vector<std::string> loadPaths;

  // Find a namespace file with the given \p name in the load path and \r retuns
  // a unique pointer to the memory buffer containing the content or an error.
  // In the success case it will put the path of the file into the \p
  // importedFile.
  ErrorOrMemBufPtr findFileInLoadPath(const std::string &name,
                                      std::string &importedFile);

  bool isValidBufferID(unsigned i) const;

  /// Converts the ns name to a partial path by replacing the dots with slashes
  static std::string convertNamespaceToPath(std::string ns_name);

public:
  SourceMgr()                  = default;
  SourceMgr(const SourceMgr &) = delete;
  SourceMgr &operator=(const SourceMgr &) = delete;
  SourceMgr(SourceMgr &&)                 = default;
  SourceMgr &operator=(SourceMgr &&) = default;
  ~SourceMgr()                       = default;

  /// Set the `loadPaths` to the given \p dirs. `loadPaths` is a vector of
  /// directories that Serene will look in order to find a file that constains a
  /// namespace which it is looking for.
  void setLoadPaths(const std::vector<std::string> &dirs) { loadPaths = dirs; }

  /// Return a reference to a `SrcBuffer` with the given ID \p i.
  const SrcBuffer &getBufferInfo(unsigned i) const {
    assert(isValidBufferID(i));
    return buffers[i - 1];
  }

  /// Return a reference to a `SrcBuffer` with the given namspace name \p ns.
  const SrcBuffer &getBufferInfo(llvm::StringRef ns) const {
    auto bufferId = nsTable.lookup(ns);

    if (bufferId == 0) {
      // No such namespace
      llvm_unreachable("couldn't find the src buffer for a namespace. It "
                       "should never happen.");
    }

    return buffers[bufferId - 1];
  }

  /// Return a pointer to the internal `llvm::MemoryBuffer` of the `SrcBuffer`
  /// with the given ID \p i.
  const llvm::MemoryBuffer *getMemoryBuffer(unsigned i) const {
    assert(isValidBufferID(i));
    return buffers[i - 1].buffer.get();
  }

  unsigned getNumBuffers() const { return buffers.size(); }

  /// Add a new source buffer to this source manager. This takes ownership of
  /// the memory buffer.
  unsigned AddNewSourceBuffer(std::unique_ptr<llvm::MemoryBuffer> f,
                              reader::LocationRange includeLoc);

  /// Lookup for a file containing the namespace definition of with given
  /// namespace name \p name. In case that the file exists, it returns an
  /// `ErrorTree`. It will use the parser to read the file and create an AST
  /// from it. Then create a namespace, set the its AST to the AST that we just
  /// read from the file and return a shared pointer to the namespace.
  ///
  /// \p importLoc is a location in the source code where the give namespace is
  /// imported.
  MaybeNS readNamespace(SereneContext &ctx, std::string name,
                        reader::LocationRange importLoc);
};

}; // namespace serene

#endif
