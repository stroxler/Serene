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

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIBSERENE_INCLUDE_DIR ${INCLUDE_DIR} PARENT_SCOPE)
include_directories(${INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_subdirectory(lib)

# Install rules for libserene target
install(TARGETS serene
  EXPORT SereneExports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


# Install rules for the public header files.
install(DIRECTORY ${INCLUDE_DIR}/serene
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN *.h
  PATTERN *.td
  PATTERN "CMake*" EXCLUDE)

# Install rule for the public generated header files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN *.h
  PATTERN *.td
  PATTERN *.h.inc
  PATTERN "CMake*" EXCLUDE)

include(CMakePackageConfigHelpers)

# Package config file let us use find_package with serene
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"

  INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/serene-${PROJECT_VERSION}
  )

write_basic_package_version_file(
  "${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/serene-${PROJECT_VERSION}
  )

# Install the package exports
install(EXPORT SereneExports
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/serene-${PROJECT_VERSION}
  NAMESPACE serene::)

# Testing only available if this is the main app
# Emergency override SERENE_CMAKE_BUILD_TESTING provided as well
if(SERENE_BUILD_TESTING)
  message("Build the test binary")
  add_subdirectory(tests)
endif()
