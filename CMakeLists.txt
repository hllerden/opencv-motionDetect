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


# target_link_libraries(MotionDetection Qt${QT_VERSION_MAJOR}::Core     ${OpenCV_LIBS}
# )

include(GNUInstallDirs)
install(TARGETS MotionDetection
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
