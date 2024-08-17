/**
* 
* Adapted from ORB_SLAM3: Examples/ROS/src/ros_rgbd.cc
*
*/

#include "common.h"

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>
#include <condition_variable>

#include <opencv2/core/core.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include "ORB_SLAM3/include/System.h"
#include "PHOTO_SLAM/include/gaussian_mapper.h"
#include "PHOTO_SLAM/viewer/imgui_viewer.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(){};

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);
    void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
    void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    if (argc > 1)
    {
        ROS_WARN ("Arguments supplied via command line are ignored.");
    }

    std::string node_name = ros::this_node::getName();

    ros::NodeHandle node_handler;
    image_transport::ImageTransport image_transport(node_handler);

    std::string voc_file, settings_file;
    node_handler.param<std::string>(node_name + "/voc_file", voc_file, "file_not_set");
    node_handler.param<std::string>(node_name + "/settings_file", settings_file, "file_not_set");

    std::string gaussian_cfg_path_, output_dir_;
    node_handler.param<std::string>(node_name + "/gaussian_cfg_path", gaussian_cfg_path_, "file_not_set");
    node_handler.param<std::string>(node_name + "/output_dir", output_dir_, "file_not_set");
    
    if (voc_file == "file_not_set" || settings_file == "file_not_set"|| gaussian_cfg_path_ == "file_not_set" || output_dir_ == "file_not_set")
    {
        ROS_ERROR("Please provide voc_file and settings_file in the launch file");       
        ros::shutdown();
        return 1;
    }

    node_handler.param<std::string>(node_name + "/world_frame_id", world_frame_id, "map");
    node_handler.param<std::string>(node_name + "/cam_frame_id", cam_frame_id, "camera");

    bool enable_pangolin;
    node_handler.param<bool>(node_name + "/enable_pangolin", enable_pangolin, true);


    bool use_viewer = true;
    node_handler.param<bool>(node_name + "/use_viewer", use_viewer, "true");

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    std::vector<float> vTimesTrack;


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    sensor_type = ORB_SLAM3::System::RGBD;
    pSLAM = new ORB_SLAM3::System(voc_file, settings_file, sensor_type, 0, "realsense_rgbd");
    //std::shared_ptr ppSLAM(pSLAM);
    //pSLAM = new std::make_shared<ORB_SLAM3::System>(voc_file, settings_file, sensor_type, 0, "realsense_rgbd");

     std::shared_ptr<ORB_SLAM3::System> ppSLAM(pSLAM);

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(gaussian_cfg_path_);
    std::filesystem::path output_dir(output_dir_);
    std::shared_ptr<GaussianMapper> pGausMapper = std::make_shared<GaussianMapper>(ppSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());
    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(ppSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    
    


    ImageGrabber igb;
    // Vector for tracking time statistics    
    vTimesTrack.reserve(3000000);


    message_filters::Subscriber<sensor_msgs::Image> sub_rgb_img(node_handler, "/camera/color/image_raw", 100);
    message_filters::Subscriber<sensor_msgs::Image> sub_depth_img(node_handler, "/camera/aligned_depth_to_color/image_raw", 100);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), sub_rgb_img, sub_depth_img);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD, &igb, _1, _2));
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    vTimesTrack.emplace_back(ttrack);


    setup_publishers(node_handler, image_transport, node_name);
    setup_services(node_handler, node_name);

    ros::spin();

    // Stop all threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

        // GPU peak usage
    igb.saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    igb.saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    ros::shutdown();

    return 0;
}

//////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
/////////////////////////

    double timestamp = cv_ptrRGB->header.stamp.toSec();
    // Pass the image to the SLAM system
    Sophus::SE3f Tcw = pSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, timestamp, std::vector<ORB_SLAM3::IMU::Point>(), std::to_string(timestamp));

/////////////////////////

    // ORB_SLAM3 runs in TrackRGBD()
    //Sophus::SE3f Tcw = pSLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, cv_ptrRGB->header.stamp.toSec());

    ros::Time msg_time = cv_ptrRGB->header.stamp;

    publish_topics(msg_time);
}

void ImageGrabber::saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    // std::sort(vTimesTrack.begin(), vTimesTrack.end());
    // out << "-------" << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "mean tracking time: " << totaltime / nImages << std::endl;

    out.close();
}

void ImageGrabber::saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}