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

set(LLVM_LINK_COMPONENTS Support TableGen)

add_executable(tbl-srn
  errors-backend.cpp
  main.cpp
)

set(SERENE_TABLEGEN "tbl-srn" CACHE
  STRING "Native TableGen executable. Saves building one when cross-compiling.")

# Effective tblgen executable to be used:
set(SERENE_TABLEGEN_EXE ${SERENE_TABLEGEN} PARENT_SCOPE)
set(SERENE_TABLEGEN_TARGET ${SERENE_TABLEGEN} PARENT_SCOPE)

target_link_libraries(tbl-srn PRIVATE LLVMTableGenGlobalISel)
set_target_properties(tbl-srn PROPERTIES FOLDER "tbl-srn")
