set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_MAN YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

set(DOXYGEN_PROJECT_BRIEF "Just another Lisp")

doxygen_add_docs(docs
  #"${CMAKE_CURRENT_SOURCE_DIR}/index.md"
  "${PROJECT_SOURCE_DIR}/src/"
  ${INCLUDE_DIR})



find_package(Doxygen REQUIRED)

# Find all the public headers
get_target_property(SERENE_PUBLIC_HEADER_DIR SERENE INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE SERENE_PUBLIC_HEADER_DIR ${SERENE_PUBLIC_HEADER_DIR}/*.h)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
# Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${SERENE_PUBLIC_HEADER_DIR}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})
add_dependencies(libserene docs)

find_package(Sphinx REQUIRED)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

add_custom_target(Sphinx ALL
                  COMMAND
                  ${SPHINX_EXECUTABLE} -b html
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating documentation with Sphinx")
