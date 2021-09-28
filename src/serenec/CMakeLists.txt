# Serene programming language.
#
#  Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_executable(serenec serenec.cpp $<TARGET_OBJECTS:Serene::lib>)

if (CPP_20_SUPPORT)
  target_compile_features(serenec PRIVATE cxx_std_20)
else()
  target_compile_features(serenec PRIVATE cxx_std_17)
endif()

get_property(serene_lib_deps TARGET Serene::lib PROPERTY StaticDeps)

target_link_libraries(serenec
  PRIVATE
  ${serene_lib_deps}
  LLVMX86AsmParser
  LLVMTarget
  LLVMOption

  clangDriver
  clangBasic
  clangdSupport
  clangFrontend
  clangEdit
  clangLex
  )

target_include_directories(serenec PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(serenec PRIVATE ${INCLUDE_DIR})

install(TARGETS serenec DESTINATION bin)
