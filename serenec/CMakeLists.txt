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

add_executable(serenec serenec.cpp)

set_target_properties(serenec PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  # Warn on unused libs
  LINK_WHAT_YOU_USE TRUE

  # LTO support
  INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_options(serenec
  PRIVATE
  $<$<CONFIG:DEBUG>:-fsanitize=address>
  $<$<CONFIG:DEBUG>:-static-libsan>
  )

target_link_options(serenec
  PRIVATE
  $<$<CONFIG:DEBUG>:-fsanitize=address>
  $<$<CONFIG:DEBUG>:-static-libsan>
  )

add_dependencies(serenec Serene::core)

if(SERENE_ENABLE_TIDY)
  set_target_properties(serenec PROPERTIES CXX_CLANG_TIDY ${CLANG_TIDY_PATH})
endif()

if (CPP_20_SUPPORT)
  target_compile_features(serenec PRIVATE cxx_std_20)
else()
  target_compile_features(serenec PRIVATE cxx_std_17)
endif()

target_link_libraries(serenec
  PRIVATE

  Serene::lib
  #MLIRPass

  #LLVMTarget
  #LLVMOption
  #lldDriver
  #lldELF
  )

target_include_directories(serenec PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(serenec PRIVATE ${INCLUDE_DIR})


install(TARGETS serenec
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
