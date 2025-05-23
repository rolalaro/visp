cmake_minimum_required(VERSION 3.10)

project(tutorial-image)

find_package(VISP REQUIRED visp_core visp_io visp_gui)

set(tutorial_cpp
  tutorial-canny-hsv.cpp
  tutorial-image-colormap.cpp
  tutorial-image-converter.cpp
  tutorial-image-manipulation.cpp
  tutorial-image-reader.cpp
  tutorial-image-viewer.cpp
  tutorial-undistort.cpp
  tutorial-viewer.cpp
  tutorial-image-filter.cpp
  tutorial-image-display.cpp
  tutorial-image-display-scaled-manu.cpp
  tutorial-image-display-scaled-auto.cpp
  tutorial-draw-line.cpp
  tutorial-draw-rectangle.cpp
  tutorial-draw-point.cpp
  tutorial-draw-circle.cpp
  tutorial-draw-cross.cpp
  tutorial-draw-text.cpp
  tutorial-draw-frame.cpp
  tutorial-export-image.cpp
  tutorial-event-keyboard.cpp
  tutorial-video-manipulation.cpp)

list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/memorial.pfm" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/monkey.bmp" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/monkey.ppm" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/monkey.pgm" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/monkey.jpeg" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/monkey.png" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/chessboard.jpg" )
list(APPEND tutorial_data "${CMAKE_CURRENT_SOURCE_DIR}/camera.xml" )

foreach(cpp ${tutorial_cpp})
  visp_add_target(${cpp})
  if(COMMAND visp_add_dependency)
    visp_add_dependency(${cpp} "tutorials")
  endif()
endforeach()

visp_add_target(tutorial-canny.cpp drawingHelpers.cpp)
if(COMMAND visp_add_dependency)
    visp_add_dependency(tutorial-canny.cpp "tutorials")
endif()

# Copy the data files to the same location than the target
foreach(data ${tutorial_data})
  visp_copy_data(tutorial-viewer.cpp ${data})
endforeach()
