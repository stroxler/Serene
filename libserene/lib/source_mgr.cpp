/*
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

#include "serene/source_mgr.h"

#include "serene/namespace.h"
#include "serene/reader/location.h"
#include "serene/reader/reader.h"
#include "serene/utils.h"

#include <system_error>

#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Locale.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LogicalResult.h>

namespace serene {

std::string SourceMgr::convertNamespaceToPath(std::string ns_name) {
  std::replace(ns_name.begin(), ns_name.end(), '.', '/');

  llvm::SmallString<MAX_PATH_SLOTS> path;
  path.append(ns_name);
  llvm::sys::path::native(path);

  return std::string(path);
};

bool SourceMgr::isValidBufferID(unsigned i) const {
  return i != 0 && i <= buffers.size();
};

SourceMgr::MemBufPtr SourceMgr::findFileInLoadPath(const std::string &name,
                                                   std::string &importedFile) {

  auto path = convertNamespaceToPath(name);

  // If the file didn't exist directly, see if it's in an include path.
  for (unsigned i = 0, e = loadPaths.size(); i != e; ++i) {

    // TODO: Ugh, Udgly, fix this using llvm::sys::path functions
    importedFile = loadPaths[i] + llvm::sys::path::get_separator().data() +
                   path + "." + DEFAULT_SUFFIX;

    SMGR_LOG("Try to load the ns from: " + importedFile);
    auto newBufOrErr = llvm::MemoryBuffer::getFile(importedFile);

    if (auto err = newBufOrErr.getError()) {
      llvm::consumeError(llvm::errorCodeToError(err));
      continue;
    }

    return std::move(*newBufOrErr);
  }

  return nullptr;
};

MaybeNS SourceMgr::readNamespace(SereneContext &ctx, std::string name,
                                 reader::LocationRange importLoc) {
  std::string importedFile;

  SMGR_LOG("Attempt to load namespace: " + name);
  MemBufPtr newBufOrErr(findFileInLoadPath(name, importedFile));

  if (newBufOrErr == nullptr) {
    auto msg = llvm::formatv("Couldn't find namespace '{0}'", name).str();
    return errors::makeError<errors::NSLoadError>(importLoc, msg);
  }

  auto bufferId = AddNewSourceBuffer(std::move(newBufOrErr), importLoc);

  UNUSED(nsTable.insert_or_assign(name, bufferId));

  if (bufferId == 0) {
    auto msg = llvm::formatv("Couldn't add namespace '{0}'", name).str();
    return errors::makeError<errors::NSAddToSMError>(importLoc, msg);
  }

  // Since we moved the buffer to be added as the source storage we
  // need to get a pointer to it again
  const auto *buf = getMemoryBuffer(bufferId);

  // Read the content of the buffer by passing it the reader
  auto maybeAst = reader::read(ctx, buf->getBuffer(), name,
                               llvm::Optional(llvm::StringRef(importedFile)));

  if (!maybeAst) {
    SMGR_LOG("Couldn't Read namespace: " + name);
    return maybeAst.takeError();
  }

  // Create the NS and set the AST
  auto ns =
      ctx.makeNamespace(name, llvm::Optional(llvm::StringRef(importedFile)));

  if (auto errs = ns->addTree(*maybeAst)) {
    SMGR_LOG("Couldn't set the AST for namespace: " + name);
    return errs;
  }

  return ns;
};

unsigned SourceMgr::AddNewSourceBuffer(std::unique_ptr<llvm::MemoryBuffer> f,
                                       reader::LocationRange includeLoc) {
  SrcBuffer nb;
  nb.buffer    = std::move(f);
  nb.importLoc = includeLoc;
  buffers.push_back(std::move(nb));
  return buffers.size();
};

template <typename T>
static std::vector<T> &GetOrCreateOffsetCache(void *&offsetCache,
                                              llvm::MemoryBuffer *buffer) {
  if (offsetCache) {
    return *static_cast<std::vector<T> *>(offsetCache);
  }

  // Lazily fill in the offset cache.
  auto *offsets = new std::vector<T>();
  size_t sz     = buffer->getBufferSize();

  // TODO: Replace this assert with a realtime check
  assert(sz <= std::numeric_limits<T>::max());

  llvm::StringRef s = buffer->getBuffer();
  for (size_t n = 0; n < sz; ++n) {
    if (s[n] == '\n') {
      offsets->push_back(static_cast<T>(n));
    }
  }

  offsetCache = offsets;
  return *offsets;
}

template <typename T>
const char *SourceMgr::SrcBuffer::getPointerForLineNumberSpecialized(
    unsigned lineNo) const {
  std::vector<T> &offsets =
      GetOrCreateOffsetCache<T>(offsetCache, buffer.get());

  // We start counting line and column numbers from 1.
  if (lineNo != 0) {
    --lineNo;
  }

  const char *bufStart = buffer->getBufferStart();

  // The offset cache contains the location of the \n for the specified line,
  // we want the start of the line.  As such, we look for the previous entry.
  if (lineNo == 0) {
    return bufStart;
  }

  if (lineNo > offsets.size()) {
    return nullptr;
  }
  return bufStart + offsets[lineNo - 1] + 1;
}

/// Return a pointer to the first character of the specified line number or
/// null if the line number is invalid.
const char *
SourceMgr::SrcBuffer::getPointerForLineNumber(unsigned lineNo) const {
  size_t sz = buffer->getBufferSize();
  if (sz <= std::numeric_limits<uint8_t>::max()) {
    return getPointerForLineNumberSpecialized<uint8_t>(lineNo);
  }

  if (sz <= std::numeric_limits<uint16_t>::max()) {
    return getPointerForLineNumberSpecialized<uint16_t>(lineNo);
  }

  if (sz <= std::numeric_limits<uint32_t>::max()) {
    return getPointerForLineNumberSpecialized<uint32_t>(lineNo);
  }

  return getPointerForLineNumberSpecialized<uint64_t>(lineNo);
}

SourceMgr::SrcBuffer::SrcBuffer(SourceMgr::SrcBuffer &&other) noexcept
    : buffer(std::move(other.buffer)), offsetCache(other.offsetCache),
      importLoc(other.importLoc) {
  other.offsetCache = nullptr;
}

SourceMgr::SrcBuffer::~SrcBuffer() {
  if (offsetCache != nullptr) {
    size_t sz = buffer->getBufferSize();
    if (sz <= std::numeric_limits<uint8_t>::max()) {
      delete static_cast<std::vector<uint8_t> *>(offsetCache);
    } else if (sz <= std::numeric_limits<uint16_t>::max()) {
      delete static_cast<std::vector<uint16_t> *>(offsetCache);
    } else if (sz <= std::numeric_limits<uint32_t>::max()) {
      delete static_cast<std::vector<uint32_t> *>(offsetCache);
    } else {
      delete static_cast<std::vector<uint64_t> *>(offsetCache);
    }
    offsetCache = nullptr;
  }
}

}; // namespace serene
