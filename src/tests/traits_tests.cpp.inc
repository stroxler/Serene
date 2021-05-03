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

#include "./test_helpers.cpp.inc"
#include "serene/traits.h"
#include <catch2/catch.hpp>

namespace serene {

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
};

class A : public WithTrait<A, Printable, Analyzable> {

  std::string x;

public:
  A(std::string a) : x(a){};
  std::string Print() const { return "A: print"; }

  std::string Analyze() const { return "A: " + x; }
};

template <typename T = FinalImpl>
class B : public std::conditional<std::is_same_v<T, FinalImpl>,
                                  WithTrait<B<>, Printable, Analyzable>,
                                  WithTrait<T, Printable, Analyzable>>::type {

  std::string y;

public:
  B(std::string a) : y(a){};
  std::string Print() const { return "B: print"; }

  std::string Analyze() const { return "B: " + y; }
};

class C : public B<C> {

  std::string z;
  std::string w;

public:
  C(std::string a, std::string b, std::string c) : B<C>(a), z(b), w(c){};
  std::string Print() const { return z; }

  std::string Analyze() const { return w; }
};

class D : public WithTrait<D, Printable> {
public:
  std::string Print() const { return "D: print"; }

  std::string Analyze() const { return "D: analyze with no trait"; }
};

template <typename T>
class MetaTrait : public WithTrait<T, Printable, Analyzable> {};

class E : public MetaTrait<E> {
public:
  std::string Print() const { return "E: print"; };

  std::string Analyze() const { return "E: in E"; };
};

TEST_CASE("Trait functionality tests", "[Traits]") {
  auto a = A("in A");
  auto b = B("in B");
  auto c = C("gray", "white", "black");
  auto d = D();
  auto e = E();

  CHECK(Print(a) == "A: print");
  CHECK(Print(b) == "B: print");
  CHECK(Print(c) == "white");
  CHECK(Print(d) == "D: print");
  CHECK(Print(e) == "E: print");

  CHECK(Analyze(a) == "A: in A");
  CHECK(Analyze(b) == "B: in B");
  CHECK(Analyze(c) == "black");
  CHECK(Analyze(e) == "E: in E");

  // Even though D has a Analyze method, It's not Analyzable to the
  // Analyze function signature won't match
  CHECK(d.Analyze() == "D: analyze with no trait");
};

} // namespace serene
