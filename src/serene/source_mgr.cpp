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

#include "mlir/Support/LogicalResult.h"
#include "serene/errors/constants.h"
#include "serene/namespace.h"
#include "serene/reader/location.h"
#include "serene/reader/reader.h"

#include "llvm/Support/MemoryBufferRef.h"

#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Locale.h>
#include <llvm/Support/Path.h>
#include <system_error>

namespace serene {
static const size_t tabStop = 8;

std::string inline SourceMgr::convertNamespaceToPath(std::string ns_name) {
  std::replace(ns_name.begin(), ns_name.end(), '.', '/');

  llvm::SmallString<256> path;
  path.append(ns_name);
  llvm::sys::path::native(path);

  return std::string(path);
};

NSPtr SourceMgr::readNamespace(SereneContext &ctx, std::string name,
                               reader::LocationRange importLoc, bool entryNS) {
  std::string includedFile;
  auto path = convertNamespaceToPath(name);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> newBufOrErr(
      std::make_error_code(std::errc::no_such_file_or_directory));

  SMGR_LOG("Attempt to load namespace: " + name);
  // If the file didn't exist directly, see if it's in an include path.
  for (unsigned i = 0, e = loadPaths.size(); i != e && !newBufOrErr; ++i) {

    // TODO: Ugh, Udgly, fix this using llvm::sys::path functions
    includedFile = loadPaths[i] + llvm::sys::path::get_separator().data() +
                   path + "." + DEFAULT_SUFFIX;

    SMGR_LOG("Try to load the ns from: " + includedFile);
    newBufOrErr = llvm::MemoryBuffer::getFile(includedFile);
  }

  if (!newBufOrErr) {
    auto msg = llvm::formatv("Couldn't find namespace '{0}'", name);
    ctx.diagEngine->emitSyntaxError(importLoc, errors::NSLoadError,
                                    llvm::StringRef(msg));
    return nullptr;
  }

  auto bufferId = AddNewSourceBuffer(std::move(*newBufOrErr), importLoc);

  if (bufferId == 0) {
    auto msg = llvm::formatv("Couldn't add namespace '{0}'", name);
    ctx.diagEngine->emitSyntaxError(importLoc, errors::NSAddToSMError,
                                    llvm::StringRef(msg));

    return nullptr;
  }

  // Since we moved the buffer to be added as the source storage we
  // need to get a pointer to it again
  auto *buf = getMemoryBuffer(bufferId);

  // Read the content of the buffer by passing it the reader
  auto maybeAst = reader::read(ctx, buf->getBuffer(), name,
                               llvm::Optional(llvm::StringRef(includedFile)));

  if (!maybeAst) {
    SMGR_LOG("Couldn't Read namespace: " + name)
    return nullptr;
  }

  // Create the NS and set the AST
  auto ns =
      makeNamespace(ctx, name, llvm::Optional(llvm::StringRef(includedFile)));

  if (mlir::failed(ns->setTree(maybeAst.getValue()))) {
    SMGR_LOG("Couldn't set the AST for namespace: " + name)
    return nullptr;
  }

  return ns;
};

unsigned SourceMgr::AddNewSourceBuffer(std::unique_ptr<llvm::MemoryBuffer> f,
                                       reader::LocationRange includeLoc) {
  SrcBuffer nb;
  nb.buffer     = std::move(f);
  nb.includeLoc = includeLoc;
  buffers.push_back(std::move(nb));
  return buffers.size();
};

// unsigned SourceMgr::AddIncludeFile(const std::string &filename,
//                                    llvm::SMLoc includeLoc,
//                                    std::string &includedFile) {
//   includedFile = filename;
//   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> NewBufOrErr =
//       llvm::MemoryBuffer::getFile(includedFile);

//   // If the file didn't exist directly, see if it's in an include path.
//   for (unsigned i = 0, e = loadPaths.size(); i != e && !NewBufOrErr; ++i) {
//     includedFile =
//         loadPaths[i] + llvm::sys::path::get_separator().data() + filename;
//     NewBufOrErr = llvm::MemoryBuffer::getFile(includedFile);
//   }

//   if (!NewBufOrErr)
//     return 0;

//   return AddNewSourceBuffer(std::move(*NewBufOrErr), includeLoc);
// }

// unsigned SourceMgr::FindBufferContainingLoc(llvm::SMLoc loc) const {
//   for (unsigned i = 0, e = buffers.size(); i != e; ++i)
//     if (loc.getPointer() >= buffers[i].buffer->getBufferStart() &&
//         // Use <= here so that a pointer to the null at the end of the buffer
//         // is included as part of the buffer.
//         loc.getPointer() <= buffers[i].buffer->getBufferEnd())
//       return i + 1;
//   return 0;
// }

// template <typename T>
// static std::vector<T> &GetOrCreateOffsetCache(void *&offsetCache,
//                                               llvm::MemoryBuffer *buffer) {
//   if (offsetCache)
//     return *static_cast<std::vector<T> *>(offsetCache);

//   // Lazily fill in the offset cache.
//   auto *offsets = new std::vector<T>();
//   size_t sz     = buffer->getBufferSize();
//   assert(sz <= std::numeric_limits<T>::max());
//   llvm::StringRef s = buffer->getBuffer();
//   for (size_t n = 0; n < sz; ++n) {
//     if (s[n] == '\n')
//       offsets->push_back(static_cast<T>(n));
//   }

//   offsetCache = offsets;
//   return *offsets;
// }

// template <typename T>
// unsigned SourceMgr::SrcBuffer::getLineNumberSpecialized(const char *ptr)
// const {
//   std::vector<T> &offsets =
//       GetOrCreateOffsetCache<T>(offsetCache, buffer.get());

//   const char *bufStart = buffer->getBufferStart();
//   assert(ptr >= bufStart && ptr <= buffer->getBufferEnd());
//   ptrdiff_t ptrDiff = ptr - bufStart;
//   assert(ptrDiff >= 0 &&
//          static_cast<size_t>(ptrDiff) <= std::numeric_limits<T>::max());
//   T ptrOffset = static_cast<T>(ptrDiff);

//   // llvm::lower_bound gives the number of EOL before PtrOffset. Add 1 to get
//   // the line number.
//   return llvm::lower_bound(offsets, ptrOffset) - offsets.begin() + 1;
// }

// /// Look up a given \p Ptr in in the buffer, determining which line it came
// /// from.
// unsigned SourceMgr::SrcBuffer::getLineNumber(const char *ptr) const {
//   size_t sz = buffer->getBufferSize();
//   if (sz <= std::numeric_limits<uint8_t>::max())
//     return getLineNumberSpecialized<uint8_t>(ptr);
//   else if (sz <= std::numeric_limits<uint16_t>::max())
//     return getLineNumberSpecialized<uint16_t>(ptr);
//   else if (sz <= std::numeric_limits<uint32_t>::max())
//     return getLineNumberSpecialized<uint32_t>(ptr);
//   else
//     return getLineNumberSpecialized<uint64_t>(ptr);
// }

// template <typename T>
// const char *SourceMgr::SrcBuffer::getPointerForLineNumberSpecialized(
//     unsigned lineNo) const {
//   std::vector<T> &offsets =
//       GetOrCreateOffsetCache<T>(offsetCache, buffer.get());

//   // We start counting line and column numbers from 1.
//   if (lineNo != 0)
//     --lineNo;

//   const char *bufStart = buffer->getBufferStart();

//   // The offset cache contains the location of the \n for the specified line,
//   // we want the start of the line.  As such, we look for the previous entry.
//   if (lineNo == 0)
//     return bufStart;
//   if (lineNo > offsets.size())
//     return nullptr;
//   return bufStart + offsets[lineNo - 1] + 1;
// }

// /// Return a pointer to the first character of the specified line number or
// /// null if the line number is invalid.
// const char *
// SourceMgr::SrcBuffer::getPointerForLineNumber(unsigned lineNo) const {
//   size_t sz = buffer->getBufferSize();
//   if (sz <= std::numeric_limits<uint8_t>::max())
//     return getPointerForLineNumberSpecialized<uint8_t>(lineNo);
//   else if (sz <= std::numeric_limits<uint16_t>::max())
//     return getPointerForLineNumberSpecialized<uint16_t>(lineNo);
//   else if (sz <= std::numeric_limits<uint32_t>::max())
//     return getPointerForLineNumberSpecialized<uint32_t>(lineNo);
//   else
//     return getPointerForLineNumberSpecialized<uint64_t>(lineNo);
// }

SourceMgr::SrcBuffer::SrcBuffer(SourceMgr::SrcBuffer &&other)
    : buffer(std::move(other.buffer)), offsetCache(other.offsetCache),
      includeLoc(other.includeLoc) {
  other.offsetCache = nullptr;
}

SourceMgr::SrcBuffer::~SrcBuffer() {
  if (offsetCache) {
    size_t sz = buffer->getBufferSize();
    if (sz <= std::numeric_limits<uint8_t>::max())
      delete static_cast<std::vector<uint8_t> *>(offsetCache);
    else if (sz <= std::numeric_limits<uint16_t>::max())
      delete static_cast<std::vector<uint16_t> *>(offsetCache);
    else if (sz <= std::numeric_limits<uint32_t>::max())
      delete static_cast<std::vector<uint32_t> *>(offsetCache);
    else
      delete static_cast<std::vector<uint64_t> *>(offsetCache);
    offsetCache = nullptr;
  }
}

// std::pair<unsigned, unsigned>
// SourceMgr::getLineAndColumn(llvm::SMLoc loc, unsigned bufferID) const {
//   if (!bufferID)
//     bufferID = FindBufferContainingLoc(loc);
//   assert(bufferID && "Invalid location!");

//   auto &sb        = getBufferInfo(bufferID);
//   const char *ptr = loc.getPointer();

//   unsigned lineNo      = sb.getLineNumber(ptr);
//   const char *bufStart = sb.buffer->getBufferStart();
//   size_t newlineOffs =
//       llvm::StringRef(bufStart, ptr - bufStart).find_last_of("\n\r");
//   if (newlineOffs == llvm::StringRef::npos)
//     newlineOffs = ~(size_t)0;
//   return std::make_pair(lineNo, ptr - bufStart - newlineOffs);
// }

// // FIXME: Note that the formatting of source locations is spread between
// // multiple functions, some in SourceMgr and some in SMDiagnostic. A better
// // solution would be a general-purpose source location formatter
// // in one of those two classes, or possibly in llvm::SMLoc.

// /// Get a string with the source location formatted in the standard
// /// style, but without the line offset. If \p IncludePath is true, the path
// /// is included. If false, only the file name and extension are included.
// std::string SourceMgr::getFormattedLocationNoOffset(llvm::SMLoc loc,
//                                                     bool includePath) const {
//   auto bufferID = FindBufferContainingLoc(loc);
//   assert(bufferID && "Invalid location!");
//   auto fileSpec = getBufferInfo(bufferID).buffer->getBufferIdentifier();

//   if (includePath) {
//     return fileSpec.str() + ":" + std::to_string(FindLineNumber(loc,
//     bufferID));
//   } else {
//     auto I = fileSpec.find_last_of("/\\");
//     I      = (I == fileSpec.size()) ? 0 : (I + 1);
//     return fileSpec.substr(I).str() + ":" +
//            std::to_string(FindLineNumber(loc, bufferID));
//   }
// }

// /// Given a line and column number in a mapped buffer, turn it into an
// /// llvm::SMLoc. This will return a null llvm::SMLoc if the line/column
// location
// /// is invalid.
// llvm::SMLoc SourceMgr::FindLocForLineAndColumn(unsigned bufferID,
//                                                unsigned lineNo,
//                                                unsigned colNo) {
//   auto &sb        = getBufferInfo(bufferID);
//   const char *ptr = sb.getPointerForLineNumber(lineNo);
//   if (!ptr)
//     return llvm::SMLoc();

//   // We start counting line and column numbers from 1.
//   if (colNo != 0)
//     --colNo;

//   // If we have a column number, validate it.
//   if (colNo) {
//     // Make sure the location is within the current line.
//     if (ptr + colNo > sb.buffer->getBufferEnd())
//       return llvm::SMLoc();

//     // Make sure there is no newline in the way.
//     if (llvm::StringRef(ptr, colNo).find_first_of("\n\r") !=
//         llvm::StringRef::npos)
//       return llvm::SMLoc();

//     ptr += colNo;
//   }

//   return llvm::SMLoc::getFromPointer(ptr);
// }

// void SourceMgr::PrintIncludeStack(llvm::SMLoc includeLoc,
//                                   llvm::raw_ostream &os) const {
//   if (includeLoc == llvm::SMLoc())
//     return; // Top of stack.

//   unsigned curBuf = FindBufferContainingLoc(includeLoc);
//   assert(curBuf && "Invalid or unspecified location!");

//   PrintIncludeStack(getBufferInfo(curBuf).includeLoc, os);

//   os << "Included from " <<
//   getBufferInfo(curBuf).buffer->getBufferIdentifier()
//      << ":" << FindLineNumber(includeLoc, curBuf) << ":\n";
// }

// SMDiagnostic SourceMgr::GetMessage(llvm::SMLoc loc, SourceMgr::DiagKind kind,
//                                    const llvm::Twine &msg,
//                                    llvm::ArrayRef<llvm::SMRange> ranges,
//                                    llvm::ArrayRef<llvm::SMFixIt> fixIts)
//                                    const {
//   // First thing to do: find the current buffer containing the specified
//   // location to pull out the source line.
//   llvm::SmallVector<std::pair<unsigned, unsigned>, 4> colRanges;
//   std::pair<unsigned, unsigned> lineAndCol;
//   llvm::StringRef bufferID = "<unknown>";
//   llvm::StringRef lineStr;

//   if (loc.isValid()) {
//     unsigned curBuf = FindBufferContainingLoc(loc);
//     assert(curBuf && "Invalid or unspecified location!");

//     const llvm::MemoryBuffer *curMB = getMemoryBuffer(curBuf);
//     bufferID                        = curMB->getBufferIdentifier();

//     // Scan backward to find the start of the line.
//     const char *lineStart = loc.getPointer();
//     const char *bufStart  = curMB->getBufferStart();
//     while (lineStart != bufStart && lineStart[-1] != '\n' &&
//            lineStart[-1] != '\r')
//       --lineStart;

//     // Get the end of the line.
//     const char *lineEnd = loc.getPointer();
//     const char *bufEnd  = curMB->getBufferEnd();
//     while (lineEnd != bufEnd && lineEnd[0] != '\n' && lineEnd[0] != '\r')
//       ++lineEnd;
//     lineStr = llvm::StringRef(lineStart, lineEnd - lineStart);

//     // Convert any ranges to column ranges that only intersect the line of
//     the
//     // location.
//     for (unsigned i = 0, e = ranges.size(); i != e; ++i) {
//       llvm::SMRange r = ranges[i];
//       if (!r.isValid())
//         continue;

//       // If the line doesn't contain any part of the range, then ignore it.
//       if (r.Start.getPointer() > lineEnd || r.End.getPointer() < lineStart)
//         continue;

//       // Ignore pieces of the range that go onto other lines.
//       if (r.Start.getPointer() < lineStart)
//         r.Start = llvm::SMLoc::getFromPointer(lineStart);
//       if (r.End.getPointer() > lineEnd)
//         r.End = llvm::SMLoc::getFromPointer(lineEnd);

//       // Translate from llvm::SMLoc ranges to column ranges.
//       // FIXME: Handle multibyte characters.
//       colRanges.push_back(std::make_pair(r.Start.getPointer() - lineStart,
//                                          r.End.getPointer() - lineStart));
//     }

//     lineAndCol = getLineAndColumn(loc, curBuf);
//   }

//   return SMDiagnostic(*this, loc, bufferID, lineAndCol.first,
//                       lineAndCol.second - 1, kind, msg.str(), lineStr,
//                       colRanges, fixIts);
// }

// void SourceMgr::PrintMessage(llvm::raw_ostream &os,
//                              const SMDiagnostic &diagnostic,
//                              bool showColors) const {
//   // Report the message with the diagnostic handler if present.
//   if (diagHandler) {
//     diagHandler(diagnostic, diagContext);
//     return;
//   }

//   if (diagnostic.getLoc().isValid()) {
//     unsigned CurBuf = FindBufferContainingLoc(diagnostic.getLoc());
//     assert(CurBuf && "Invalid or unspecified location!");
//     PrintIncludeStack(getBufferInfo(CurBuf).includeLoc, os);
//   }

//   diagnostic.print(nullptr, os, showColors);
// }

// void SourceMgr::PrintMessage(llvm::raw_ostream &os, llvm::SMLoc loc,
//                              SourceMgr::DiagKind kind, const llvm::Twine
//                              &msg, llvm::ArrayRef<llvm::SMRange> ranges,
//                              llvm::ArrayRef<llvm::SMFixIt> fixIts,
//                              bool showColors) const {
//   PrintMessage(os, GetMessage(loc, kind, msg, ranges, fixIts), showColors);
// }

// void SourceMgr::PrintMessage(llvm::SMLoc loc, SourceMgr::DiagKind kind,
//                              const llvm::Twine &msg,
//                              llvm::ArrayRef<llvm::SMRange> ranges,
//                              llvm::ArrayRef<llvm::SMFixIt> fixIts,
//                              bool showColors) const {
//   PrintMessage(llvm::errs(), loc, kind, msg, ranges, fixIts, showColors);
// }

//===----------------------------------------------------------------------===//
// SMDiagnostic Implementation
//===----------------------------------------------------------------------===//

SMDiagnostic::SMDiagnostic(const SourceMgr &sm, llvm::SMLoc l,
                           llvm::StringRef fn, int line, int col,
                           SourceMgr::DiagKind kind, llvm::StringRef msg,
                           llvm::StringRef lineStr,
                           llvm::ArrayRef<std::pair<unsigned, unsigned>> ranges,
                           llvm::ArrayRef<llvm::SMFixIt> hints)
    : sm(&sm), loc(l), filename(std::string(fn)), lineNo(line), columnNo(col),
      kind(kind), message(msg), lineContents(lineStr), ranges(ranges.vec()),
      fixIts(hints.begin(), hints.end()) {
  llvm::sort(fixIts);
}

static void buildFixItLine(std::string &caretLine, std::string &fixItLine,
                           llvm::ArrayRef<llvm::SMFixIt> fixIts,
                           llvm::ArrayRef<char> sourceLine) {
  if (fixIts.empty())
    return;

  const char *lineStart = sourceLine.begin();
  const char *lineEnd   = sourceLine.end();

  size_t prevHintEndCol = 0;

  for (const llvm::SMFixIt &fixit : fixIts) {
    // If the fixit contains a newline or tab, ignore it.
    if (fixit.getText().find_first_of("\n\r\t") != llvm::StringRef::npos)
      continue;

    llvm::SMRange r = fixit.getRange();

    // If the line doesn't contain any part of the range, then ignore it.
    if (r.Start.getPointer() > lineEnd || r.End.getPointer() < lineStart)
      continue;

    // Translate from llvm::SMLoc to column.
    // Ignore pieces of the range that go onto other lines.
    // FIXME: Handle multibyte characters in the source line.
    unsigned firstCol;
    if (r.Start.getPointer() < lineStart)
      firstCol = 0;
    else
      firstCol = r.Start.getPointer() - lineStart;

    // If we inserted a long previous hint, push this one forwards, and add
    // an extra space to show that this is not part of the previous
    // completion. This is sort of the best we can do when two hints appear
    // to overlap.
    //
    // Note that if this hint is located immediately after the previous
    // hint, no space will be added, since the location is more important.
    unsigned hintCol = firstCol;
    if (hintCol < prevHintEndCol)
      hintCol = prevHintEndCol + 1;

    // FIXME: This assertion is intended to catch unintended use of multibyte
    // characters in fixits. If we decide to do this, we'll have to track
    // separate byte widths for the source and fixit lines.
    assert((size_t)llvm::sys::locale::columnWidth(fixit.getText()) ==
           fixit.getText().size());

    // This relies on one byte per column in our fixit hints.
    unsigned lastColumnModified = hintCol + fixit.getText().size();
    if (lastColumnModified > fixItLine.size())
      fixItLine.resize(lastColumnModified, ' ');

    llvm::copy(fixit.getText(), fixItLine.begin() + hintCol);

    prevHintEndCol = lastColumnModified;

    // For replacements, mark the removal range with '~'.
    // FIXME: Handle multibyte characters in the source line.
    unsigned lastCol;
    if (r.End.getPointer() >= lineEnd)
      lastCol = lineEnd - lineStart;
    else
      lastCol = r.End.getPointer() - lineStart;

    std::fill(&caretLine[firstCol], &caretLine[lastCol], '~');
  }
}

static void printSourceLine(llvm::raw_ostream &s,
                            llvm::StringRef lineContents) {
  // Print out the source line one character at a time, so we can expand tabs.
  for (unsigned i = 0, e = lineContents.size(), outCol = 0; i != e; ++i) {
    size_t nextTab = lineContents.find('\t', i);
    // If there were no tabs left, print the rest, we are done.
    if (nextTab == llvm::StringRef::npos) {
      s << lineContents.drop_front(i);
      break;
    }

    // Otherwise, print from i to NextTab.
    s << lineContents.slice(i, nextTab);
    outCol += nextTab - i;
    i = nextTab;

    // If we have a tab, emit at least one space, then round up to 8 columns.
    do {
      s << ' ';
      ++outCol;
    } while ((outCol % tabStop) != 0);
  }
  s << '\n';
}

static bool isNonASCII(char c) { return c & 0x80; }

void SMDiagnostic::print(const char *progName, llvm::raw_ostream &os,
                         bool showColors, bool showKindLabel) const {
  llvm::ColorMode mode =
      showColors ? llvm::ColorMode::Auto : llvm::ColorMode::Disable;

  {
    llvm::WithColor s(os, llvm::raw_ostream::SAVEDCOLOR, true, false, mode);

    if (progName && progName[0])
      s << progName << ": ";

    if (!filename.empty()) {
      if (filename == "-")
        s << "<stdin>";
      else
        s << filename;

      if (lineNo != -1) {
        s << ':' << lineNo;
        if (columnNo != -1)
          s << ':' << (columnNo + 1);
      }
      s << ": ";
    }
  }

  if (showKindLabel) {
    switch (kind) {
    case SourceMgr::DK_Error:
      llvm::WithColor::error(os, "", !showColors);
      break;
    case SourceMgr::DK_Warning:
      llvm::WithColor::warning(os, "", !showColors);
      break;
    case SourceMgr::DK_Note:
      llvm::WithColor::note(os, "", !showColors);
      break;
    case SourceMgr::DK_Remark:
      llvm::WithColor::remark(os, "", !showColors);
      break;
    }
  }

  llvm::WithColor(os, llvm::raw_ostream::SAVEDCOLOR, true, false, mode)
      << message << '\n';

  if (lineNo == -1 || columnNo == -1)
    return;

  // FIXME: If there are multibyte or multi-column characters in the source, all
  // our ranges will be wrong. To do this properly, we'll need a byte-to-column
  // map like Clang's TextDiagnostic. For now, we'll just handle tabs by
  // expanding them later, and bail out rather than show incorrect ranges and
  // misaligned fixits for any other odd characters.
  if (llvm::any_of(lineContents, isNonASCII)) {
    printSourceLine(os, lineContents);
    return;
  }
  size_t numColumns = lineContents.size();

  // Build the line with the caret and ranges.
  std::string caretLine(numColumns + 1, ' ');

  // Expand any ranges.
  for (const std::pair<unsigned, unsigned> &r : ranges)
    std::fill(&caretLine[r.first],
              &caretLine[std::min((size_t)r.second, caretLine.size())], '~');

  // Add any fix-its.
  // FIXME: Find the beginning of the line properly for multibyte characters.
  std::string fixItInsertionLine;
  buildFixItLine(
      caretLine, fixItInsertionLine, fixIts,
      llvm::makeArrayRef(loc.getPointer() - columnNo, lineContents.size()));

  // Finally, plop on the caret.
  if (unsigned(columnNo) <= numColumns)
    caretLine[columnNo] = '^';
  else
    caretLine[numColumns] = '^';

  // ... and remove trailing whitespace so the output doesn't wrap for it.  We
  // know that the line isn't completely empty because it has the caret in it at
  // least.
  caretLine.erase(caretLine.find_last_not_of(' ') + 1);

  printSourceLine(os, lineContents);

  {
    llvm::ColorMode mode =
        showColors ? llvm::ColorMode::Auto : llvm::ColorMode::Disable;
    llvm::WithColor s(os, llvm::raw_ostream::GREEN, true, false, mode);

    // Print out the caret line, matching tabs in the source line.
    for (unsigned i = 0, e = caretLine.size(), outCol = 0; i != e; ++i) {
      if (i >= lineContents.size() || lineContents[i] != '\t') {
        s << caretLine[i];
        ++outCol;
        continue;
      }

      // Okay, we have a tab.  Insert the appropriate number of characters.
      do {
        s << caretLine[i];
        ++outCol;
      } while ((outCol % tabStop) != 0);
    }
    s << '\n';
  }

  // Print out the replacement line, matching tabs in the source line.
  if (fixItInsertionLine.empty())
    return;

  for (size_t i = 0, e = fixItInsertionLine.size(), outCol = 0; i < e; ++i) {
    if (i >= lineContents.size() || lineContents[i] != '\t') {
      os << fixItInsertionLine[i];
      ++outCol;
      continue;
    }

    // Okay, we have a tab.  Insert the appropriate number of characters.
    do {
      os << fixItInsertionLine[i];
      // FIXME: This is trying not to break up replacements, but then to re-sync
      // with the tabs between replacements. This will fail, though, if two
      // fix-it replacements are exactly adjacent, or if a fix-it contains a
      // space. Really we should be precomputing column widths, which we'll
      // need anyway for multibyte chars.
      if (fixItInsertionLine[i] != ' ')
        ++i;
      ++outCol;
    } while (((outCol % tabStop) != 0) && i != e);
  }
  os << '\n';
}

}; // namespace serene
