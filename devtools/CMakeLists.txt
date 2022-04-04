# Serene Programming Language
#
# Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
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

if (SERENE_ENABLE_DEVTOOLS)

  add_executable(slir-lsp-server slir-lsp-server.cpp)
add_executable(Serene::SLIR::LSP ALIAS slir-lsp-server)

set_target_properties(slir-lsp-server PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE

  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)

if(SERENE_ENABLE_TIDY)
  set_target_properties(slir-lsp-server PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
endif()

if (CPP_20_SUPPORT)
  target_compile_features(slir-lsp-server PRIVATE cxx_std_20)
else()
  target_compile_features(slir-lsp-server PRIVATE cxx_std_17)
endif()


target_link_libraries(slir-lsp-server
  PRIVATE
  Serene::lib
  MLIRLspServerLib
)

target_include_directories(slir-lsp-server PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(slir-lsp-server PRIVATE ${INCLUDE_DIR})

install(TARGETS slir-lsp-server
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


# ========
# slir-opt
# ========
add_executable(slir-opt slir-opt.cpp)
add_executable(Serene::SLIR::Opt ALIAS slir-opt)

set_target_properties(slir-opt PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE

  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)

if(SERENE_ENABLE_TIDY)
  set_target_properties(slir-opt PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
endif()

if (CPP_20_SUPPORT)
  target_compile_features(slir-opt PRIVATE cxx_std_20)
else()
  target_compile_features(slir-opt PRIVATE cxx_std_17)
endif()


target_link_libraries(slir-opt
  PRIVATE
  Serene::lib
  MLIROptLib
)

target_include_directories(slir-opt PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(slir-opt PRIVATE ${INCLUDE_DIR})

install(TARGETS slir-opt
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()
