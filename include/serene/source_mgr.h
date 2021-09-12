/* -*- C++ -*-
 * Serene programming language.
 *
 *  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef SERENE_SOURCE_MGR_H
#define SERENE_SOURCE_MGR_H

#include "serene/namespace.h"
#include "serene/reader/location.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <memory>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/Timing.h>

#define SMGR_LOG(...)                       \
  DEBUG_WITH_TYPE("sourcemgr", llvm::dbgs() \
                                   << "[SMGR]: " << __VA_ARGS__ << "\n");

namespace serene {
class SereneContext;
class SMDiagnostic;

class SourceMgr {

public:
  std::string DEFAULT_SUFFIX = "srn";

  enum DiagKind {
    DK_Error,
    DK_Warning,
    DK_Remark,
    DK_Note,
  };

  /// Clients that want to handle their own diagnostics in a custom way can
  /// register a function pointer+context as a diagnostic handler.
  /// It gets called each time PrintMessage is invoked.
  using DiagHandlerTy = void (*)(const SMDiagnostic &, void *context);

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

    /// This is the location of the parent include, or null if at the top level.
    reader::LocationRange includeLoc;

    SrcBuffer() = default;
    SrcBuffer(SrcBuffer &&);
    SrcBuffer(const SrcBuffer &) = delete;
    SrcBuffer &operator=(const SrcBuffer &) = delete;
    ~SrcBuffer();
  };

  /// This is all of the buffers that we are reading from.
  std::vector<SrcBuffer> buffers;

  llvm::StringMap<unsigned> nsTable;

  /// A mapping from the ns name to buffer id. The ns name is a reference to
  /// the actual name that is stored in the Namespace instance.
  llvm::DenseMap<llvm::StringRef, unsigned> nsToBufId;

  // This is the list of directories we should search for include files in.
  std::vector<std::string> loadPaths;

  DiagHandlerTy diagHandler = nullptr;
  void *diagContext         = nullptr;

  bool isValidBufferID(unsigned i) const { return i && i <= buffers.size(); }

  /// Converts the ns name to a partial path by replacing the dots with slashes
  std::string inline convertNamespaceToPath(std::string ns_name);

public:
  SourceMgr()                  = default;
  SourceMgr(const SourceMgr &) = delete;
  SourceMgr &operator=(const SourceMgr &) = delete;
  SourceMgr(SourceMgr &&)                 = default;
  SourceMgr &operator=(SourceMgr &&) = default;
  ~SourceMgr()                       = default;

  void setLoadPaths(const std::vector<std::string> &dirs) { loadPaths = dirs; }

  /// Specify a diagnostic handler to be invoked every time PrintMessage is
  /// called. \p Ctx is passed into the handler when it is invoked.
  void setDiagHandler(DiagHandlerTy dh, void *ctx = nullptr) {
    diagHandler = dh;
    diagContext = ctx;
  }

  DiagHandlerTy getDiagHandler() const { return diagHandler; }
  void *getDiagContext() const { return diagContext; }

  const SrcBuffer &getBufferInfo(unsigned i) const {
    assert(isValidBufferID(i));
    return buffers[i - 1];
  }

  const SrcBuffer &getBufferInfo(llvm::StringRef ns) const {
    auto bufferId = nsTable.lookup(ns);

    if (bufferId == 0) {
      // No such namespace
      llvm_unreachable("couldn't find the src buffer for a namespace. It "
                       "should never happen.");
    }

    return buffers[bufferId - 1];
  }

  const llvm::MemoryBuffer *getMemoryBuffer(unsigned i) const {
    assert(isValidBufferID(i));
    return buffers[i - 1].buffer.get();
  }

  unsigned getNumBuffers() const { return buffers.size(); }

  unsigned getMainFileID() const {
    assert(getNumBuffers());
    return 1;
  }

  // reader::LocationRange getParentIncludeLoc(unsigned i) const {
  //   assert(isValidBufferID(i));
  //   return buffers[i - 1].includeLoc;
  // }

  /// Add a new source buffer to this source manager. This takes ownership of
  /// the memory buffer.
  unsigned AddNewSourceBuffer(std::unique_ptr<llvm::MemoryBuffer> f,
                              reader::LocationRange includeLoc);
  /// Search for a file with the specified name in the current directory or in
  /// one of the IncludeDirs.
  ///
  /// If no file is found, this returns 0, otherwise it returns the buffer ID
  /// of the stacked file. The full path to the included file can be found in
  /// \p IncludedFile.
  // unsigned AddIncludeFile(const std::string &filename, llvm::SMLoc
  // includeLoc,
  //                         std::string &includedFile);

  NSPtr readNamespace(SereneContext &ctx, std::string name,
                      reader::LocationRange importLoc, bool entryNS = false);

  // /// Return the ID of the buffer containing the specified location.
  // ///
  // /// 0 is returned if the buffer is not found.
  // unsigned FindBufferContainingLoc(llvm::SMLoc loc) const;

  // /// Find the line number for the specified location in the specified file.
  // /// This is not a fast method.
  // unsigned FindLineNumber(llvm::SMLoc loc, unsigned bufferID = 0) const {
  //   return getLineAndColumn(loc, bufferID).first;
  // }

  // /// Find the line and column number for the specified location in the
  // /// specified file. This is not a fast method.
  // std::pair<unsigned, unsigned> getLineAndColumn(llvm::SMLoc loc,
  //                                                unsigned bufferID = 0)
  //                                                const;

  // /// Get a string with the \p llvm::SMLoc filename and line number
  // /// formatted in the standard style.
  // std::string getFormattedLocationNoOffset(llvm::SMLoc loc,
  //                                          bool includePath = false) const;

  // /// Given a line and column number in a mapped buffer, turn it into an
  // /// llvm::SMLoc. This will return a null llvm::SMLoc if the line/column
  // /// location is invalid.
  // llvm::SMLoc FindLocForLineAndColumn(unsigned bufferID, unsigned lineNo,
  //                                     unsigned colNo);

  // /// Emit a message about the specified location with the specified string.
  // ///
  // /// \param ShowColors Display colored messages if output is a terminal and
  // /// the default error handler is used.
  // void PrintMessage(llvm::raw_ostream &os, llvm::SMLoc loc, DiagKind kind,
  //                   const llvm::Twine &msg,
  //                   llvm::ArrayRef<llvm::SMRange> ranges = {},
  //                   llvm::ArrayRef<llvm::SMFixIt> fixIts = {},
  //                   bool showColors                      = true) const;

  // /// Emits a diagnostic to llvm::errs().
  // void PrintMessage(llvm::SMLoc loc, DiagKind kind, const llvm::Twine &msg,
  //                   llvm::ArrayRef<llvm::SMRange> ranges = {},
  //                   llvm::ArrayRef<llvm::SMFixIt> fixIts = {},
  //                   bool showColors                      = true) const;

  // /// Emits a manually-constructed diagnostic to the given output stream.
  // ///
  // /// \param ShowColors Display colored messages if output is a terminal and
  // /// the default error handler is used.
  // void PrintMessage(llvm::raw_ostream &os, const SMDiagnostic &diagnostic,
  //                   bool showColors = true) const;

  // /// Return an SMDiagnostic at the specified location with the specified
  // /// string.
  // ///
  // /// \param Msg If non-null, the kind of message (e.g., "error") which is
  // /// prefixed to the message.
  // SMDiagnostic GetMessage(llvm::SMLoc loc, DiagKind kind,
  //                         const llvm::Twine &msg,
  //                         llvm::ArrayRef<llvm::SMRange> ranges = {},
  //                         llvm::ArrayRef<llvm::SMFixIt> fixIts = {}) const;

  // /// Prints the names of included files and the line of the file they were
  // /// included from. A diagnostic handler can use this before printing its
  // /// custom formatted message.
  // ///
  // /// \param IncludeLoc The location of the include.
  // /// \param OS the raw_ostream to print on.
  // void PrintIncludeStack(llvm::SMLoc includeLoc, llvm::raw_ostream &os)
  // const;
};

/// Instances of this class encapsulate one diagnostic report, allowing
/// printing to a raw_ostream as a caret diagnostic.
class SMDiagnostic {
  const SourceMgr *sm = nullptr;
  llvm::SMLoc loc;
  std::string filename;
  int lineNo               = 0;
  int columnNo             = 0;
  SourceMgr::DiagKind kind = SourceMgr::DK_Error;
  std::string message, lineContents;
  std::vector<std::pair<unsigned, unsigned>> ranges;
  llvm::SmallVector<llvm::SMFixIt, 4> fixIts;

public:
  // Null diagnostic.
  SMDiagnostic() = default;
  // Diagnostic with no location (e.g. file not found, command line arg error).
  SMDiagnostic(llvm::StringRef filename, SourceMgr::DiagKind knd,
               llvm::StringRef msg)
      : filename(filename), lineNo(-1), columnNo(-1), kind(knd), message(msg) {}

  // Diagnostic with a location.
  SMDiagnostic(const SourceMgr &sm, llvm::SMLoc l, llvm::StringRef fn, int line,
               int col, SourceMgr::DiagKind kind, llvm::StringRef msg,
               llvm::StringRef lineStr,
               llvm::ArrayRef<std::pair<unsigned, unsigned>> ranges,
               llvm::ArrayRef<llvm::SMFixIt> fixIts = {});

  const SourceMgr *getSourceMgr() const { return sm; }
  llvm::SMLoc getLoc() const { return loc; }
  llvm::StringRef getFilename() const { return filename; }
  int getLineNo() const { return lineNo; }
  int getColumnNo() const { return columnNo; }
  SourceMgr::DiagKind getKind() const { return kind; }
  llvm::StringRef getMessage() const { return message; }
  llvm::StringRef getLineContents() const { return lineContents; }
  llvm::ArrayRef<std::pair<unsigned, unsigned>> getRanges() const {
    return ranges;
  }

  void addFixIt(const llvm::SMFixIt &hint) { fixIts.push_back(hint); }

  llvm::ArrayRef<llvm::SMFixIt> getFixIts() const { return fixIts; }

  void print(const char *progName, llvm::raw_ostream &s, bool showColors = true,
             bool showKindLabel = true) const;
};

}; // namespace serene

#endif
