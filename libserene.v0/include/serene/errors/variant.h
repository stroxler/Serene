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

#ifndef SERENE_ERRORS_VARIANT_H
#define SERENE_ERRORS_VARIANT_H

#include <string>

namespace serene::errors {

// This class is used in the generated code
struct ErrorVariant {
  const int id;
  const std::string title;
  const std::string desc;
  const std::string help;

  static ErrorVariant make(const int id, const char *t, const char *d,
                           const char *h) {
    return ErrorVariant(id, t, d, h);
  };

private:
  ErrorVariant(const int id, const char *t, const char *d, const char *h)
      : id(id), title(t), desc(d), help(h){};
};
} // namespace serene::errors
#endif
