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

#ifndef READER_H
#define READER_H

#include <fmt/core.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "serene/expr.hpp"
#include "serene/list.hpp"
#include "serene/logger.hpp"
#include "serene/serene.hpp"
#include "serene/symbol.hpp"

#if defined(ENABLE_READER_LOG) || defined(ENABLE_LOG)
#define READER_LOG(...) __LOG("READER", __VA_ARGS__);
#else
#define READER_LOG(...) ;
#endif

namespace serene {

class ReadError : public std::exception {
private:
  char *message;

public:
  ReadError(char *msg) : message(msg){};
  const char *what() const throw() { return message; }
};

class Reader {
private:
  std::stringstream input_stream;

  char get_char(bool);
  void unget_char();
  bool is_valid_for_identifier(char);

  // The property to store the ast tree
  ast_tree ast;

  ast_node read_symbol();
  ast_list_node read_list(List *);
  ast_node read_expr();

public:
  Reader(const std::string &);
  ast_tree &read();

  ~Reader();
};
} // namespace serene

#endif
