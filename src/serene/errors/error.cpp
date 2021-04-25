/*
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

#include "serene/errors/error.h"
#include "llvm/Support/FormatVariadic.h"

namespace serene {
namespace errors {

serene::exprs::ExprType Error::getType() const {
  return serene::exprs::ExprType::Error;
};

std::string Error::toString() const {
  return llvm::formatv("<Error E{0}: {1}>", this->variant->id, this->message);
}

serene::exprs::MaybeNode Error::analyze(SereneContext &ctx) {
  return Result<serene::exprs::Node>::success(nullptr);
};

bool Error::classof(const serene::exprs::Expression *e) {
  return e->getType() == serene::exprs::ExprType::Error;
};

} // namespace errors
} // namespace serene
