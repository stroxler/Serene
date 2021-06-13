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

/**
 * This is a CRTP based Trait implementation that allows to use Trait like
 * classes to create polymorphic functions and API statically at compile type
 * without any runtime shenanigans. For more on CRTP checkout:
 *
 * https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 *
 * In order to define a trait, use the `TraitBase` class like:

 * \code
 * template <typename ConcreteType>
 * class Blahable : public TraitBase<ConcreteType, Blahable> {}
 * \endcode
 *
 * Every Trait has to take the `ConcreteType` as template argument and pass it
 * to the `TraitBase`. Checkout the documentation of `TraitBase` for more info
 * on creating a new Trait.
 *
 * Alongside with each Trait type, you should provide the "Official" interface
 * of the Trait via some standalone functions that operates on the Trait type.
 * For example Imagine we have a Trait type called `ABC` with two main
 * functionality `foo` and `bar`. We need to create two functions as follows:
 *
 * \code
 * template <typename T>
 * SomeType Foo(ABC<T> &t) { return t.foo(); };
 *
 * template <typename T>
 * SomeType bar(ABC<T> &t, int x) { return t.bar(x); };
 * \endcode
 *
 * These two functions will be the official interface to the trait `ABC`.
 * IMPORTANT NOTE: Make sure to pass a reference of type `ABC<T>` to the
 * functions and DO NOT PASS BY COPY. Since copying will copy the value by the
 * trait type only, we would not be able to statically case it to the
 * implementor type and it will lead to some allocation problem.
 *
 * Traits can be used via `WithTrait` helper class which provides a clean
 * interface to mix and match Trait types.
 *
 */
#ifndef SERENE_TRAITS_H
#define SERENE_TRAITS_H

#include <string>
#include <type_traits>

namespace serene {

/// A placeholder structure that replaces the concrete type of the
/// Imlementations Which might have child classes.
struct FinalImpl;

/// In order to use Traits, we can use `WithTrait` class as the base
/// of any implementation class and pass the Trait classes as template argument
/// for example:
///
/// \code
/// class Expression : public WithTrait<Expression, Printable, Locatable> {}
/// \endcode
template <typename ConcreteType, template <typename T> class... Traits>
class WithTrait : public Traits<ConcreteType>... {
protected:
  WithTrait(){};
  friend ConcreteType;
};

/// This class provides the common functionality among the Trait Types and
/// every Trait has to inherit from this class. Here is an example:
///
/// \code
/// template <typename ConcreteType>
/// class Blahable : public TraitBase<ConcreteType, Blahable> {}
/// \endcode
///
/// In the Trait class the underlaying object which implements the Trait
/// is accessable via the `Object` method.
template <typename ConcreteType, template <typename> class TraitType>
class TraitBase {
protected:
  /// Statically casts the object to the concrete type object to be
  /// used in the Trait Types.
  // const ConcreteType &Object() const {
  //   return static_cast<const ConcreteType &>(*this);
  // };

  ConcreteType &Object() { return static_cast<ConcreteType &>(*this); };
};

template <typename ConcreteType>
class IDebuggable : public TraitBase<ConcreteType, IDebuggable> {
public:
  IDebuggable(){};
  IDebuggable(const IDebuggable &) = delete;
  std::string toString() const { return this->Object().toString(); }
};

template <typename T>
std::string toString(IDebuggable<T> &t) {
  return t.toString();
}

}; // namespace serene
#endif
