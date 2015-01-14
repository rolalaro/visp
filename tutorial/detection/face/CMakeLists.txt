project(tutorial-detection-face)

cmake_minimum_required(VERSION 2.6)

find_package(VISP REQUIRED)

# set the list of source files
set(tutorial_cpp
  tutorial-face-detector-live.cpp)

list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_frontalface_alt.xml" )

foreach(cpp ${tutorial_cpp})
  visp_add_target(${cpp})
  if(COMMAND visp_add_dependency)
    visp_add_dependency(${cpp} "tutorials")
  endif()
endforeach()

# Copy the data files to the same location than the target
foreach(data ${tutorial_data})
  visp_copy_data(tutorial-face-detector-live.cpp ${data})
endforeach()
