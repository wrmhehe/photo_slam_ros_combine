cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo_slam_ros)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CAFFE2_USE_CUDNN on)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)



find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  message_filters
  roscpp
  rospy
  tf
  tf2
  message_generation
)
##
#set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/libtorch)
set(CMAKE_PREFIX_PATH /home/wrm/Pictures/libtorch)
set(Torch_DIR /home/wrm/Pictures/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

find_package(CUDA REQUIRED)
find_package(PCL)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(realsense2)

add_service_files(
  FILES
  SaveMap.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package (
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport tf sensor_msgs dynamic_reconfigure message_runtime
  LIBRARIES {PROJECT_NAME} libDBoW2 libg2o
)



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/include
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/include/CameraModels
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/Sophus
  ${PROJECT_SOURCE_DIR}/PHOTO_SLAM
  ${PROJECT_SOURCE_DIR}/PHOTO_SLAM/include
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


# include(${PROJECT_SOURCE_DIR}/ORB_SLAM3/CMakeLists.txt)
# include(${PROJECT_SOURCE_DIR}/PHOTO_SLAM/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/DBoW2/CMakeLists.txt)
include(${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/g2o/CMakeLists.txt)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


add_library(${PROJECT_NAME} SHARED
  ORB_SLAM3/src/System.cc
  ORB_SLAM3/src/Tracking.cc
  ORB_SLAM3/src/LocalMapping.cc
  ORB_SLAM3/src/LoopClosing.cc
  ORB_SLAM3/src/ORBextractor.cc
  ORB_SLAM3/src/ORBmatcher.cc
  ORB_SLAM3/src/FrameDrawer.cc
  ORB_SLAM3/src/Converter.cc
  ORB_SLAM3/src/MapPoint.cc
  ORB_SLAM3/src/KeyFrame.cc
  ORB_SLAM3/src/Atlas.cc
  ORB_SLAM3/src/Map.cc
  ORB_SLAM3/src/MapDrawer.cc
  ORB_SLAM3/src/Optimizer.cc
  ORB_SLAM3/src/Frame.cc
  ORB_SLAM3/src/KeyFrameDatabase.cc
  ORB_SLAM3/src/Sim3Solver.cc
  ORB_SLAM3/src/Viewer.cc
  ORB_SLAM3/src/ImuTypes.cc
  ORB_SLAM3/src/G2oTypes.cc
  ORB_SLAM3/src/CameraModels/Pinhole.cpp
  ORB_SLAM3/src/CameraModels/KannalaBrandt8.cpp
  ORB_SLAM3/src/OptimizableTypes.cpp
  ORB_SLAM3/src/MLPnPsolver.cpp
  ORB_SLAM3/src/GeometricTools.cc
  ORB_SLAM3/src/TwoViewReconstruction.cc
  ORB_SLAM3/src/Config.cc
  ORB_SLAM3/src/Settings.cc
  ORB_SLAM3/src/PointCloudMapper.cpp
  
  ORB_SLAM3/include/System.h
  ORB_SLAM3/include/Tracking.h
  ORB_SLAM3/include/LocalMapping.h
  ORB_SLAM3/include/LoopClosing.h
  ORB_SLAM3/include/ORBextractor.h
  ORB_SLAM3/include/ORBmatcher.h
  ORB_SLAM3/include/FrameDrawer.h
  ORB_SLAM3/include/Converter.h
  ORB_SLAM3/include/MapPoint.h
  ORB_SLAM3/include/KeyFrame.h
  ORB_SLAM3/include/Atlas.h
  ORB_SLAM3/include/Map.h
  ORB_SLAM3/include/MapDrawer.h
  ORB_SLAM3/include/Optimizer.h
  ORB_SLAM3/include/Frame.h
  ORB_SLAM3/include/KeyFrameDatabase.h
  ORB_SLAM3/include/Sim3Solver.h
  ORB_SLAM3/include/Viewer.h
  ORB_SLAM3/include/ImuTypes.h
  ORB_SLAM3/include/G2oTypes.h
  ORB_SLAM3/include/CameraModels/GeometricCamera.h
  ORB_SLAM3/include/CameraModels/Pinhole.h
  ORB_SLAM3/include/CameraModels/KannalaBrandt8.h
  ORB_SLAM3/include/OptimizableTypes.h
  ORB_SLAM3/include/MLPnPsolver.h
  ORB_SLAM3/include/GeometricTools.h
  ORB_SLAM3/include/TwoViewReconstruction.h
  ORB_SLAM3/include/SerializationUtils.h
  ORB_SLAM3/include/Config.h
  ORB_SLAM3/include/Settings.h
  ORB_SLAM3/include/PointCloudMapper.h
)
 
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/g2o/lib/libg2o.so
  ${TORCH_LIBRARIES}
  ${PCL_LIBRARIES}
  #${PROJECT_SOURCE_DIR}/ORB_SLAM3/lib/libORB_SLAM3.so 
  #${PROJECT_SOURCE_DIR}/PHOTO_SLAM/lib/libPHOTO_SLAM.so 
  -lboost_system
  -lboost_serialization
  -lcrypto

)




add_library(simple_knn SHARED
    PHOTO_SLAM/third_party/simple-knn/simple_knn.cu
    PHOTO_SLAM/third_party/simple-knn/simple_knn.h
    PHOTO_SLAM/third_party/simple-knn/spatial.cu
    PHOTO_SLAM/third_party/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    PHOTO_SLAM/include/operate_points.h
    PHOTO_SLAM/src/operate_points.cu
    PHOTO_SLAM/include/rasterize_points.h
    PHOTO_SLAM/src/rasterize_points.cu
    PHOTO_SLAM/include/stereo_vision.h
    PHOTO_SLAM/src/stereo_vision.cu
    PHOTO_SLAM/cuda_rasterizer/auxiliary.h
    PHOTO_SLAM/cuda_rasterizer/backward.cu
    PHOTO_SLAM/cuda_rasterizer/backward.h
    PHOTO_SLAM/cuda_rasterizer/config.h
    PHOTO_SLAM/cuda_rasterizer/forward.cu
    PHOTO_SLAM/cuda_rasterizer/forward.h
    PHOTO_SLAM/cuda_rasterizer/operate_points.h
    PHOTO_SLAM/cuda_rasterizer/rasterizer.h
    PHOTO_SLAM/cuda_rasterizer/rasterizer_impl.cu
    PHOTO_SLAM/cuda_rasterizer/rasterizer_impl.h
    PHOTO_SLAM/cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui SHARED
    PHOTO_SLAM/viewer/imgui/imconfig.h
    PHOTO_SLAM/viewer/imgui/imgui_demo.cpp
    PHOTO_SLAM/viewer/imgui/imgui_draw.cpp
    PHOTO_SLAM/viewer/imgui/imgui_impl_glfw.cpp
    PHOTO_SLAM/viewer/imgui/imgui_impl_glfw.h
    PHOTO_SLAM/viewer/imgui/imgui_impl_opengl3_loader.h
    PHOTO_SLAM/viewer/imgui/imgui_impl_opengl3.cpp
    PHOTO_SLAM/viewer/imgui/imgui_impl_opengl3.h
    PHOTO_SLAM/viewer/imgui/imgui_internal.h
    PHOTO_SLAM/viewer/imgui/imgui_tables.cpp
    PHOTO_SLAM/viewer/imgui/imgui_widgets.cpp
    PHOTO_SLAM/viewer/imgui/imgui.cpp
    PHOTO_SLAM/viewer/imgui/imgui.h
    PHOTO_SLAM/viewer/imgui/imstb_rectpack.h
    PHOTO_SLAM/viewer/imgui/imstb_textedit.h
    PHOTO_SLAM/viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    PHOTO_SLAM/third_party/tinyply/tinyply.h
    PHOTO_SLAM/third_party/tinyply/tinyply.cpp
    PHOTO_SLAM/include/gaussian_keyframe.h
    PHOTO_SLAM/include/gaussian_model.h
    PHOTO_SLAM/include/gaussian_parameters.h
    PHOTO_SLAM/include/gaussian_rasterizer.h
    PHOTO_SLAM/include/gaussian_renderer.h
    PHOTO_SLAM/include/gaussian_scene.h
    PHOTO_SLAM/include/gaussian_trainer.h
    PHOTO_SLAM/include/gaussian_mapper.h
    PHOTO_SLAM/include/general_utils.h
    PHOTO_SLAM/include/graphics_utils.h
    PHOTO_SLAM/include/loss_utils.h
    PHOTO_SLAM/include/sh_utils.h
    PHOTO_SLAM/include/tensor_utils.h
    PHOTO_SLAM/include/camera.h
    PHOTO_SLAM/include/point_cloud.h
    PHOTO_SLAM/include/point2d.h
    PHOTO_SLAM/include/point3d.h
    PHOTO_SLAM/include/types.h
    PHOTO_SLAM/src/gaussian_keyframe.cpp
    PHOTO_SLAM/src/gaussian_model.cpp
    PHOTO_SLAM/src/gaussian_parameters.cpp
    PHOTO_SLAM/src/gaussian_rasterizer.cpp
    PHOTO_SLAM/src/gaussian_renderer.cpp
    PHOTO_SLAM/src/gaussian_scene.cpp
    PHOTO_SLAM/src/gaussian_trainer.cpp
    PHOTO_SLAM/src/gaussian_mapper.cpp)
target_link_libraries(gaussian_mapper
    ${PROJECT_NAME} 
    #${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer)

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer SHARED
    PHOTO_SLAM/viewer/drawer_utils.h
    PHOTO_SLAM/viewer/imgui_viewer.cpp
    PHOTO_SLAM/viewer/imgui_viewer.h
    PHOTO_SLAM/viewer/map_drawer.cpp
    PHOTO_SLAM/viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${PROJECT_NAME}
    #${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# # This is a C++ libtorch implementation of gaussian-splatting (https://github.com/graphdeco-inria/gaussian-splatting)
# add_executable(train_colmap PHOTO_SLAM/examples/train_colmap.cpp)
# target_link_libraries(train_colmap
#     gaussian_viewer
#     gaussian_mapper)

# add_executable(view_result PHOTO_SLAM/examples/view_result.cpp)
# target_link_libraries(view_result
#     gaussian_viewer
#     gaussian_mapper)

# ##################################################################################
# ##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
# ##################################################################################

# # Replica Monocular
# add_executable(replica_mono PHOTO_SLAM/examples/replica_mono.cpp)
# target_link_libraries(replica_mono
#     gaussian_viewer    
#     gaussian_mapper
#     ${PROJECT_NAME})

# # Replica Monocular
# add_executable(replica_rgbd PHOTO_SLAM/examples/replica_rgbd.cpp)
# target_link_libraries(replica_rgbd
#     gaussian_viewer    
#     gaussian_mapper
#     ${PROJECT_NAME})

# # TUM Monocular
# add_executable(tum_mono PHOTO_SLAM/examples/tum_mono.cpp)
# target_link_libraries(tum_mono
#     gaussian_viewer    
#     gaussian_mapper
#     ${PROJECT_NAME})

# # TUM RGBD
# add_executable(tum_rgbd PHOTO_SLAM/examples/tum_rgbd.cpp)
# target_link_libraries(tum_rgbd
#     gaussian_viewer    
#     gaussian_mapper
#     ${PROJECT_NAME})

# # EuRoC Stereo
# add_executable(euroc_stereo PHOTO_SLAM/examples/euroc_stereo.cpp)
# target_link_libraries(euroc_stereo
#     gaussian_viewer    
#     gaussian_mapper
#     ${PROJECT_NAME})

# ##################################################################################
# ##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
# ##################################################################################

# # Intel Realsense
# if(realsense2_FOUND)
#     add_executable(realsense_rgbd PHOTO_SLAM/examples/realsense_rgbd.cpp)
    
#     target_include_directories(realsense_rgbd PUBLIC
#         ${realsense_INCLUDE_DIR})
        
#     target_link_libraries(realsense_rgbd
#         gaussian_viewer    
#         gaussian_mapper
#         ${PROJECT_NAME}
#         ${realsense2_LIBRARY}
#         ${OpenCV_LIBRARIES})
# endif()






# ## ROS node
# add_executable(ros_mono
#   src/ros_mono.cc
#   src/common.cc
# )
# target_link_libraries(ros_mono
#   ${PROJECT_NAME}
#   #${catkin_LIBRARIES}
# )

# ## ROS node
# add_executable(ros_mono_inertial 
#   src/ros_mono_inertial.cc
#   src/common.cc
# )
# target_link_libraries(ros_mono_inertial 
#   ${PROJECT_NAME}
#   #${catkin_LIBRARIES}
# )

# ## ROS node
# add_executable(ros_stereo 
#   src/ros_stereo.cc
#   src/common.cc
# )
# target_link_libraries(ros_stereo
#   ${PROJECT_NAME}
#   #${catkin_LIBRARIES}
# )

# ## ROS node
# add_executable(ros_stereo_inertial
#   src/ros_stereo_inertial.cc
#   src/common.cc
# )
# target_link_libraries(ros_stereo_inertial 
#   ${PROJECT_NAME}
#   #${catkin_LIBRARIES}
# )

## ROS node
add_executable(ros_rgbd
  src/ros_rgbd.cc
  src/common.cc
)
target_link_libraries(ros_rgbd
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  gaussian_viewer    
  gaussian_mapper
  ${OpenCV_LIBRARIES})

# ## ROS node
# add_executable(ros_rgbd_inertial
#   src/ros_rgbd_inertial.cc
#   src/common.cc
# )
# target_link_libraries(ros_rgbd_inertial
#   ${PROJECT_NAME}
#   ${catkin_LIBRARIES}
# )
