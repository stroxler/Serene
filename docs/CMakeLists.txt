set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_MAN YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

set(DOXYGEN_PROJECT_BRIEF "Just another Lisp")

message("<<<<<<<<<<<<<<<<<<<<<<<<<<<")
message(${PROJECT_SOURCE_DIR})
message(${INCLUDE_DIR})
doxygen_add_docs(docs
  #"${CMAKE_CURRENT_SOURCE_DIR}/index.md"
  "${PROJECT_SOURCE_DIR}/src/"
  ${INCLUDE_DIR})

add_dependencies(serenec docs)
