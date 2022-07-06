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


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${LIBSERENE_INCLUDE_DIR})

add_subdirectory(lib/serene)


get_target_property(LIBNAME Serene::core CMAKE_PKG_NAME)

# Install rules for libserene target
install(TARGETS core
  EXPORT CoreExports
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
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${LIBNAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}Config.cmake"

  INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/serene-core-${PROJECT_VERSION}
  )

write_basic_package_version_file(
  "CoreConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}ConfigVersion.cmake"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/serene-core-${PROJECT_VERSION}
  )

# Install the package exports
install(EXPORT CoreExports
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/serene-core-${PROJECT_VERSION}
  NAMESPACE serene::)

# Testing only available if this is the main app
# Emergency override SERENE_CMAKE_BUILD_TESTING provided as well
if(SERENE_BUILD_TESTING)
  message("Build the test binary")
  add_subdirectory(tests)
endif()
