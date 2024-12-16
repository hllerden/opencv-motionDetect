cmake_minimum_required(VERSION 3.16)

project(MotionDetection LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
# find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})


add_executable(MotionDetection
  main.cpp
)
target_link_libraries(MotionDetection      ${OpenCV_LIBS}
)

add_executable(histogram
    histogram.cpp
)
target_link_libraries(histogram      ${OpenCV_LIBS}
)
add_executable(edgeDetection
    edgeDetection.cpp
)
target_link_libraries(edgeDetection      ${OpenCV_LIBS}
)
add_executable(objDetect
    objDetect.cpp
)
target_link_libraries(objDetect      ${OpenCV_LIBS}
)




# target_link_libraries(MotionDetection Qt${QT_VERSION_MAJOR}::Core     ${OpenCV_LIBS}
# )

include(GNUInstallDirs)
install(TARGETS MotionDetection
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS histogram
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS edgeDetection
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS objDetect
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
