cmake_minimum_required(VERSION 3.0.2)
project(wall_follower_rl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  gazebo_msgs
)

catkin_package()

## Install executable scripts
catkin_install_python(PROGRAMS
  scripts/q_learning_node.py
  scripts/sarsa_node.py       # uncomment when ready
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Install q_tables directory so the package ships with pre-trained tables
install(DIRECTORY q_tables/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/q_tables
)
