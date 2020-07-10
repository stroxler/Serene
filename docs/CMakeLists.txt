set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
doxygen_add_docs(docs
  serene/serene.hpp
  "${CMAKE_CURRENT_SOURCE_DIR}/index.md"
  WORKING_DIRECTORY
  "${INCLUDE_DIR}")
