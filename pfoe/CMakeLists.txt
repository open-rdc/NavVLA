cmake_minimum_required(VERSION 3.8)
project(pfoe)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# --- pfoe core library ---
add_library(pfoe_core
  src/Event.cpp
  src/Episode.cpp
  src/Particle.cpp
  src/ParticleFilter.cpp
)
target_include_directories(pfoe_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(pfoe_core PUBLIC cxx_std_17)

# --- pfoe_node executable ---
add_executable(pfoe_node src/pfoe_node.cpp)
target_include_directories(pfoe_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(pfoe_node PRIVATE cxx_std_17)
target_link_libraries(pfoe_node pfoe_core)
ament_target_dependencies(pfoe_node rclcpp std_msgs)

# --- install ---
install(TARGETS pfoe_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(PROGRAMS
  scripts/clip_encoder_node.py
  scripts/record_episode.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
