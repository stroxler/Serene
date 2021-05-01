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

#ifndef TRAITS_TRAIT_H
#define TRAITS_TRAIT_H

#include <string>
#include <type_traits>

struct FinalImpl;

template <typename ConcreteType, template <typename T> class... Traits>
class WithTrait : public Traits<ConcreteType>... {
protected:
  WithTrait(){};
  friend ConcreteType;
};

template <typename ConcreteType, template <typename> class TraitType>
class TraitBase {
protected:
  const ConcreteType &Object() const {
    return static_cast<const ConcreteType &>(*this);
  };
};

template <typename ConcreteType>
class Printable : public TraitBase<ConcreteType, Printable> {
public:
  Printable(){};
  Printable(const Printable &) = delete;
  std::string Print() const { return this->Object().Print(); }
};

template <typename ConcreteType>
class Analyzable : public TraitBase<ConcreteType, Analyzable> {
public:
  Analyzable(){};
  Analyzable(const Analyzable &) = delete;
  std::string Analyze() const { return this->Object().Analyze(); }
};

template <typename T> std::string Print(Printable<T> &t) { return t.Print(); }

template <typename T> std::string Analyze(Analyzable<T> &t) {
  return t.Analyze();
}

#endif
