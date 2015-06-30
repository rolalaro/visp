#vp_define_module(core)

# Add optional 3rd parties
set(opt_incs "")
set(opt_libs "")

# winmm.lib for timeGetTime() under windows
CHECK_LIBRARY_EXISTS("winmm.lib" getch "" HAVE_LIBWINMM) # for timeGetTime()
if(HAVE_LIBWINMM)
  list(APPEND opt_libs "winmm.lib")
endif()

# Add library ws2_32.a or ws2_32.lib for vpNetwork class
if(WIN32 AND NOT CYGWIN)
  if(MINGW)
    CHECK_LIBRARY_EXISTS("ws2_32.a" getch "" HAVE_LIBWS2_32) # for inet_ntoa() and socket functionalities
    if(HAVE_LIBWS2_32)
      list(APPEND opt_libs "ws2_32.a")
    else()
      find_library(WS2_32_LIBRARY "libws2_32.a"
        "$ENV{MINGW_DIR}/lib"
        "$ENV{MINGW_DIR}/mingw/lib"
        C:/mingw/mingw/lib)
      mark_as_advanced(WS2_32_LIBRARY)
      if(WS2_32_LIBRARY)
        list(APPEND opt_libs "${WS2_32_LIBRARY}")
      endif()
    endif()
  else() # pure WIN32
    CHECK_LIBRARY_EXISTS("ws2_32.lib" getch "" HAVE_LIBWS2_32) # for inet_ntoa() and socket functionalities
    if(HAVE_LIBWS2_32)
      #message("have ws2_32.lib")
      list(APPEND opt_libs "ws2_32.lib")
    endif()
  endif()
endif()

# OpenCV
if(USE_OPENCV)
  # On win32 since OpenCV 2.4.7 and on OSX with OpenCV 2.4.10 we cannot use OpenCV_LIBS to set ViSP 3rd party libraries.
  # Using OpenCV_LIBS works to build visp library, examples, demos and test thanks to the components,
  # but not tutorials that are stand alone Cmake project that use ViSP as a 3rd party.
  # To be clear OpenCV_LIBS contains opencv_core and not c:\...\opencv_core248.lib full path as requested
  # to use ViSP. This was not the case with OpenCV 2.4.6.
  # For the build of ViSP it works with OpenCV_LIBS: in that case thanks to opencv_core properties, CMake
  # is able to find the real name and location of the libraries.
  # But when ViSP is used as a 3rd party where it should import OpenCV libraries, it doesn't work with
  # OpenCV_LIBS.
  # The solution here is to get the real name of OpenCV libraries thanks to the properties and link
  # with these names.
  # An other way could be to include OpenCVConfig.cmake, but in that case, visp-config and visp.pc
  # will be not able to give the names of OpenCV libraries when used without CMake.
  #message("OpenCV_LIB_COMPONENTS: ${OpenCV_LIB_COMPONENTS}")
  #message("OpenCV_LIBS: ${OpenCV_LIBS}")
  #if(WIN32 AND OpenCV_LIB_COMPONENTS AND OpenCV_VERSION AND OpenCV_VERSION VERSION_GREATER 2.4.6.1)
  if(OpenCV_LIB_COMPONENTS AND OpenCV_VERSION AND OpenCV_VERSION VERSION_GREATER 2.4.6.1)
    # ReleaseWithDebugInfo was requested to detect libopencv-devel under Fedora 20
    # RelWithDebugInfo was requested to detect ros-hydro-opencv2 2.4.9 under Ubuntu 12.04 LTS with ROS hydro
    set(config_ "NONE" "RELEASE" "DEBUG" "RELEASEWITHDEBINFO" "RELWITHDEBINFO")
    if(POLICY CMP0045)
      # Fix Error on non-existent target in get_target_property for 3rd party location extraction
      cmake_policy(PUSH)
      cmake_policy(SET CMP0045 OLD)
    endif()

    foreach(component_ ${OpenCV_LIB_COMPONENTS})
      foreach(imp_config_ ${config_})
        if(OpenCV_SHARED)
          get_target_property(component_property_${imp_config_}_ ${component_} IMPORTED_IMPLIB_${imp_config_})
          # particular case of opencv_ts that doesn't have an implib
          if(NOT EXISTS "${component_property_${imp_config_}_}")
            get_target_property(component_property_${imp_config_}_ ${component_} IMPORTED_LOCATION_${imp_config_})
          endif()
        else()
          get_target_property(component_property_${imp_config_}_ ${component_} IMPORTED_LOCATION_${imp_config_})
        endif()
        get_target_property(component_property_3rdparty_${imp_config_}_ ${component_} IMPORTED_LINK_INTERFACE_LIBRARIES_${imp_config_})
        #message("component_property_${imp_config_}_: ${component_property_${imp_config_}_}")
        #message("component_property_3rdparty_${imp_config_}_: ${component_property_3rdparty_${imp_config_}_}")
        # Under Unix, there is no specific suffix for OpenCV libraries. If one is found we add it
        # Under Windows, we add the "optimized", "debug" specific keywords
        if(WIN32 AND EXISTS "${component_property_${imp_config_}_}" AND "${imp_config_}" MATCHES "RELEASE") # also valid for RELEASEWITHDEBINFO
          list(APPEND opt_libs optimized "${component_property_${imp_config_}_}")
        elseif(WIN32 AND EXISTS "${component_property_${imp_config_}_}" AND "${imp_config_}" MATCHES "DEBUG")
          list(APPEND opt_libs debug     "${component_property_${imp_config_}_}")
        elseif(EXISTS "${component_property_${imp_config_}_}")
          list(APPEND opt_libs "${component_property_${imp_config_}_}")
        endif()

        if(component_property_3rdparty_${imp_config_}_)
          foreach(3rdparty_ ${component_property_3rdparty_${imp_config_}_})
            #message("3rdparty_ ${3rdparty_}")
            list(FIND OpenCV_LIB_COMPONENTS ${3rdparty_} 3rdparty_is_opencv_component_)
            if(3rdparty_is_opencv_component_ LESS 0)
              #message("${3rdparty_} is not an opencv component")
              get_target_property(3rdparty_opt_location_ ${3rdparty_} IMPORTED_LOCATION_${imp_config_})
              if(NOT EXISTS "${3rdparty_opt_location_}")
                #message("3rdparty_: ${3rdparty_} location doesn't exist in ${imp_config_}")
                get_target_property(3rdparty_opt_location_ ${3rdparty_} IMPORTED_LOCATION)
                #message("3rdparty_: ${3rdparty_} location : ${3rdparty_opt_location_}")
              endif()
              if(EXISTS "${3rdparty_opt_location_}")
                #message("3rdparty_opt_location_: ${3rdparty_opt_location_} with config ${imp_config_}")
                if(WIN32 AND "${imp_config_}" MATCHES "RELEASE")
                  #message("is release")
                  list(APPEND opt_libs optimized ${3rdparty_opt_location_})
                elseif(WIN32 AND "${imp_config_}" MATCHES "DEBUG")
                  list(APPEND opt_libs debug ${3rdparty_opt_location_})
                else()
                  list(APPEND opt_libs ${3rdparty_opt_location_})
                endif()
              else()
                find_library(3rdparty_location_ NAMES ${3rdparty_})
                mark_as_advanced(3rdparty_location_)
                if(3rdparty_location_)
                  #message(${3rdparty_location_})
                  list(APPEND opt_libs ${3rdparty_location_}) # should be a system dependency
                else()
                  list(APPEND opt_libs ${3rdparty_}) # should be a system dependency
                endif()
              endif()
            endif()
          endforeach()
        endif()
      endforeach()
    endforeach()

    if(POLICY CMP0045)
      # Fix Error on non-existent target in get_target_property for 3rd party location extraction
      cmake_policy(POP)
    endif()

  else()
    # this should be an old OpenCV version that doesn't have the previous behavior
    list(APPEND opt_libs ${OpenCV_LIBS})
  endif()
  list(APPEND opt_incs ${OpenCV_INCLUDE_DIRS})
  #list(APPEND opt_libs ${OpenCV_LIBS})
  #MESSAGE("OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
  #MESSAGE("OpenCV_LIBS = ${OpenCV_LIBS}")
  #MESSAGE("OpenCV_LIB_DIR = ${OpenCV_LIB_DIR}")
  #message("OpenCV_NONFREE_FOUND: ${OPENCV_NONFREE_FOUND}")
endif(USE_OPENCV)


# Display X11, GTK, GDI, D3D9, OpenCV
if(USE_X11)
  list(APPEND opt_incs ${X11_INCLUDE_DIR})
  list(APPEND opt_libs ${X11_LIBRARIES})
endif()
if(USE_GTK2)
  list(APPEND opt_incs ${GTK2_INCLUDE_DIRS})
  list(APPEND opt_libs ${GTK2_LIBRARIES})
  # FS next lines removed until they proved to be useful
#  if(UNIX)
#    ## ignore gtk pedantic warnings by gcc specific -isystem :
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GTK2_gtk_INCLUDE_PATH}")
#  endif()
endif()
if(USE_GDI)
  list(APPEND opt_libs ${GDI_LIBRARIES})
endif()
if(USE_DIRECT3D)
  list(APPEND opt_incs ${DIRECT3D_INCLUDE_DIRS})
  list(APPEND opt_libs ${DIRECT3D_LIBRARIES})
endif()

# Image i/o: jpeg, png, Yarp, OpenCV
if(USE_JPEG)
  list(APPEND opt_incs ${JPEG_INCLUDE_DIR})
  list(APPEND opt_libs ${JPEG_LIBRARIES})
  # If Qt3 and libjpeg are used, we add here QT_CLEAN_NAMESPACE define to
  # handle a compilation error "conflicting declaration 'typedef long int INT32'
  if(VISP_HAVE_QT3_FOUND)
    add_definitions("-DQT_CLEAN_NAMESPACE")
  endif()
endif()
if(USE_PNG)
  list(APPEND opt_incs ${PNG_INCLUDE_DIR})
  list(APPEND opt_libs ${PNG_LIBRARIES})
  add_definitions(${PNG_DEFINITIONS})
endif()
if(USE_YARP)
  list(APPEND opt_incs ${YARP_INCLUDE_DIRS})

  # Work around to add Yarp libraries and also third party libraries requested by Yarp
  list(REVERSE YARP_LIBRARIES) # to start with YARP_init, that depends on YARP_dev, YARP_sig and YARP_OS
  foreach(lib ${YARP_LIBRARIES})
    get_target_property(CONFIGURATIONS ${lib} IMPORTED_CONFIGURATIONS)
    foreach(CONFIGURATION ${CONFIGURATIONS})
      get_target_property(YARP_LIB ${lib} "IMPORTED_LOCATION_${CONFIGURATION}") # Get Yarp full absolute library path and name

      if(WIN32)
        #Work around when YARP is build as shared libraries
        string(REGEX REPLACE ".dll$" ".lib" YARP_LIB ${YARP_LIB})

        if(${CONFIGURATION} STREQUAL "RELEASE")
          list(APPEND VISP_EXTERN_LIBRARIES optimized ${YARP_LIB}) # Append full absolute library path and name
        elseif(${CONFIGURATION} STREQUAL "DEBUG")
          list(APPEND VISP_EXTERN_LIBRARIES debug ${YARP_LIB}) # Append full absolute library path and name
        endif()
      else()
        list(APPEND opt_libs ${YARP_LIB}) # Append full absolute library path and name
      endif()
      # Get 3rd party libraries requested by Yarp
      get_target_property(YARP_LINK_LIBS_ ${lib} "IMPORTED_LINK_INTERFACE_LIBRARIES_${CONFIGURATION}")
      list(APPEND YARP_LINK_LIBS ${YARP_LINK_LIBS_})
    endforeach()
  endforeach()

  # Remove Yarp libraries since they were added previously with full absolute library path and name
  if(YARP_LINK_LIBS)
    foreach(lib ${YARP_LIBRARIES})
      list(REMOVE_ITEM YARP_LINK_LIBS ${lib})
    endforeach()
  endif()
  # Add 3rd party libraries requested by Yarp
  list(APPEND opt_libs ${YARP_LINK_LIBS})
  add_definitions(${YARP_DEFINES})
endif(USE_YARP)

# Math: gsl, lapack, OpenCV
if(USE_GSL)
  list(APPEND opt_incs ${GSL_INCLUDE_DIRS})
  list(APPEND opt_libs ${GSL_LIBRARIES})
endif()
if(USE_LAPACK)
  list(APPEND opt_libs ${LAPACK_C_LIBRARIES})
endif()

# Misc: xml, pthread, zlib
if(USE_XML2)
  list(APPEND opt_incs ${XML2_INCLUDE_DIRS})
  list(APPEND opt_libs ${XML2_LIBRARIES})
endif()
if(USE_PTHREAD)
  list(APPEND opt_incs ${PTHREAD_INCLUDE_DIRS})
  list(APPEND opt_libs ${PTHREAD_LIBRARIES})
endif()
if(USE_ZLIB)
  list(APPEND opt_incs ${ZLIB_INCLUDE_DIRS})
  list(APPEND opt_libs ${ZLIB_LIBRARIES})
endif()

# videos: ffmpeg
if(USE_FFMPEG)
  list(APPEND opt_incs ${FFMPEG_INCLUDE_DIRS})
  list(APPEND opt_libs ${FFMPEG_LIBRARIES})
endif()

vp_add_module(core)
vp_glob_module_sources()
vp_module_include_directories(${opt_incs})
vp_create_module(${opt_libs})
vp_create_compat_headers("include/visp3/core/vpConfig.h")
vp_add_tests(CTEST_EXCLUDE_PATH network)