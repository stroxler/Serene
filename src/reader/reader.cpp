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

#include "serene/reader/reader.hpp"
#include "serene/error.hpp"
#include "serene/list.hpp"
#include "serene/symbol.hpp"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include <assert.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace std;

namespace serene {

namespace reader {
Reader::Reader(const string input) { this->setInput(input); };

/**
 * Set the input of the reader.
 * @param input Set the input to the given string
 */
void Reader::setInput(const string input) {
  input_stream.write(input.c_str(), input.size());
};

Reader::~Reader() { READER_LOG("Destroying the reader"); }

/**
 * Return the next character in the buffer.
 * @param skip_whitespace If true it will skip whitespaces and EOL chars
 */
char Reader::get_char(bool skip_whitespace) {
  for (;;) {
    char c = input_stream.get();

    this->current_char = c;
    inc_location(current_location, c == '\n');

    if (skip_whitespace == true && isspace(c)) {

      continue;
    } else {
      return c;
    }
  }
};

void Reader::unget_char() {
  input_stream.unget();
  // The char that we just unget
  dec_location(current_location, this->current_char == '\n');
};

bool Reader::is_valid_for_identifier(char c) {
  switch (c) {
  case '!' | '$' | '%' | '&' | '*' | '+' | '-' | '.' | '~' | '/' | ':' | '<' |
      '=' | '>' | '?' | '@' | '^' | '_':
    return true;
  }

  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9')) {
    return true;
  }
  return false;
}

ast_node Reader::read_symbol() {
  bool empty = true;
  char c = get_char(false);

  READER_LOG("Reading symbol");
  if (!this->is_valid_for_identifier(c)) {

    // TODO: Replece this with a tranceback function or something to raise
    // synatx error.
    fmt::print("Invalid character at the start of a symbol: '{}'\n", c);
    exit(1);
  }

  std::string sym("");
  LocationRange loc;
  loc.start = current_location;

  while (c != EOF && ((!(isspace(c)) && this->is_valid_for_identifier(c)))) {
    sym += c;
    c = get_char(false);
    empty = false;
  }

  if (!empty) {
    unget_char();
    loc.end = current_location;
    return makeSymbol(loc, sym);
  }

  // TODO: it should never happens
  return nullptr;
};
// std::unique_ptr<List> list
ast_list_node Reader::read_list() {
  auto list = makeList(current_location);

  char c = get_char(true);
  assert(c == '(');

  bool list_terminated = false;

  do {
    char c = get_char(true);

    switch (c) {
    case EOF:
      throw ReadError(const_cast<char *>("EOF reached before closing of list"));
    case ')':
      list_terminated = true;
      list->location->end = current_location;

      break;

    default:
      unget_char();
      list->append(read_expr());
    }

  } while (!list_terminated);

  return list;
}

ast_node Reader::read_expr() {
  char c = get_char(false);
  READER_LOG("CHAR: {}", c);

  unget_char();

  switch (c) {
  case '(': {

    return read_list();
  }
  case EOF:
    return nullptr;

  default:
    return read_symbol();
  }
}

std::unique_ptr<ast_tree> Reader::read() {
  char c = get_char(true);

  while (c != EOF) {
    unget_char();
    auto tmp{read_expr()};
    if (tmp) {
      this->ast.push_back(move(tmp));
    }
    c = get_char(true);
  }

  return std::make_unique<ast_tree>(this->ast);
}

void Reader::dumpAST() {
  ast_tree ast = *this->read();
  std::string result = "";
  for (auto &node : ast) {
    result = fmt::format("{0} {1}", result, node->dumpAST());
  }
}

std::unique_ptr<ast_tree> FileReader::read() {

  // TODO: Add support for relative path as well
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(file);

  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    llvm::errs() << fmt::format("File: '{}'\n", file);
    llvm::errs() << "Use absolute path for now\n";
    return nullptr;
  }

  reader->setInput(fileOrErr.get()->getBuffer().str());
  return reader->read();
}

void FileReader::dumpAST() {
  auto maybeAst = this->read();
  ast_tree ast;

  if (maybeAst) {
    ast = *maybeAst;
  }

  std::string result = "";
  for (auto &node : ast) {
    result = fmt::format("{0} {1}", result, node->dumpAST());
  }
  cout << result << endl;
}

FileReader::~FileReader() {
  delete this->reader;
  READER_LOG("Destroying the file reader");
}
} // namespace reader
} // namespace serene
