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

#ifndef SERENE_ERRORS_TRAITS_H
#define SERENE_ERRORS_TRAITS_H

#include "serene/errors/constants.h"
#include "serene/traits.h"

namespace serene::errors {
template <typename ConcreteType>
class IError : public TraitBase<ConcreteType, IError> {
public:
  IError(){};
  IError(const IError &) = delete;

  ErrorVariant *getVariant();
  std::string getMessage();
};

} // namespace serene::errors
#endif
