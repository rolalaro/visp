cmake_minimum_required(VERSION 3.10)

project(tutorial-detection-object)

find_package(VISP REQUIRED visp_core visp_vision visp_mbt visp_io visp_gui)

# set the list of source files
set(tutorial_cpp
  tutorial-detection-object-mbt.cpp
  tutorial-detection-object-mbt2.cpp
  tutorial-detection-object-mbt-deprecated.cpp
  tutorial-detection-object-mbt2-deprecated.cpp
)

list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/detection-config.xml")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/detection-config-SIFT.xml")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.xml")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.cao")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.wrl")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.init")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.png")
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/teabox.mp4")

list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube.xml")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube.cao")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube.wrl")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube.init")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube.mp4")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube0001.png")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube0150.png")
list(APPEND tutorial_data2 "${CMAKE_CURRENT_SOURCE_DIR}/cube0200.png")

foreach(cpp ${tutorial_cpp})
  visp_add_target(${cpp})

  if(COMMAND visp_add_dependency)
    visp_add_dependency(${cpp} "tutorials")
  endif()
endforeach()

# Copy the data files to the same location than the target
foreach(data ${tutorial_data})
  visp_copy_data(tutorial-detection-object-mbt.cpp ${data})
endforeach()

foreach(data2 ${tutorial_data2})
  visp_copy_data(tutorial-detection-object-mbt2.cpp ${data2})
endforeach()
