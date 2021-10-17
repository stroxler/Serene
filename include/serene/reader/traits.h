/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
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

#ifndef SERENE_READER_TRAITS_H
#define SERENE_READER_TRAITS_H

#include "serene/reader/location.h"
#include "serene/traits.h"

namespace serene::reader {

template <typename ConcreteType>
class ILocatable : public TraitBase<ConcreteType, ILocatable> {
public:
  ILocatable(){};
  ILocatable(const ILocatable &) = delete;
  serene::reader::LocationRange &where() const {
    return this->Object().where();
  }
};

template <typename T>
serene::reader::LocationRange &where(ILocatable<T> &);
} // namespace serene::reader
#endif
