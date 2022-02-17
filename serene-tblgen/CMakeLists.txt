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

include_directories(${PROJECT_SOURCE_DIR}/serene-tblgen)

add_executable(serene-tblgen
  serene/errors-backend.cpp
  main.cpp
)

add_executable(SereneTablegen ALIAS serene-tblgen)

set(SERENE_TABLEGEN $<TARGET_FILE:serene-tblgen> CACHE
  STRING "Native TableGen executable. Saves building one when cross-compiling.")

set(SERENE_TABLEGEN_EXE SereneTablegen
  CACHE STRING "Where to find the tbl-srn binary")
set(SERENE_TABLEGEN_TARGET SereneTablegen CACHE
  STRING "Target name for the tbl-srn")

target_link_libraries(serene-tblgen PRIVATE LLVMTableGenGlobalISel ${llvm_libs})
set_target_properties(serene-tblgen PROPERTIES FOLDER "serene-tblgen")
