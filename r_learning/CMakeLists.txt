cmake_minimum_required(VERSION 3.0.2)
project(r_learning)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp rospy std_msgs
    geometry_msgs nav_msgs visualization_msgs
    tf2 tf2_ros tf2_geometry_msgs 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(differential_control src/differential_control.cpp)
target_link_libraries(differential_control ${catkin_LIBRARIES})

add_executable(rl_miguel src/rl_miguel.cpp)
target_link_libraries(rl_miguel ${catkin_LIBRARIES})
add_dependencies(rl_miguel ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})