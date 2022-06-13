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

#include "serene/traits.h"

#include "./test_helpers.cpp.inc"
#include <catch2/catch_all.hpp>

namespace serene {

template <typename ConcreteType>
class Printable : public TraitBase<ConcreteType, Printable> {
public:
  Printable(){};
  Printable(const Printable &) = delete;
  std::string Print() { return this->Object().Print(); }
};

template <typename ConcreteType>
class Analyzable : public TraitBase<ConcreteType, Analyzable> {
public:
  Analyzable(){};
  Analyzable(const Analyzable &) = delete;
  std::string Analyze() { return this->Object().Analyze(); }
};

template <typename T>
std::string Print(Printable<T> &t) {
  return t.Print();
}

template <typename T>
std::string Analyze(Analyzable<T> &t) {
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
