# Serene Programming Language
#
# Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

add_executable(serene-repl serene-repl.cpp)
add_executable(Serene::Repl ALIAS serene-repl)

set_target_properties(serene-repl PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE

  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)


if (CPP_20_SUPPORT)
  target_compile_features(serene-repl PRIVATE cxx_std_20)
else()
  target_compile_features(serene-repl PRIVATE cxx_std_17)
endif()


target_link_libraries(serene-repl
  PRIVATE
  Serene::lib
)

target_include_directories(serene-repl PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(serene-repl PRIVATE ${INCLUDE_DIR})

install(TARGETS serene-repl DESTINATION bin)
