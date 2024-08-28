//! \example tutorial-mb-generic-tracker-rgbd-blender.cpp
#include <iostream>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpUnscentedKalmanPose.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/mbt/vpMbGenericTracker.h>

#if defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGCODECS) && defined(VISP_HAVE_PUGIXML)
#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

namespace
{
bool read_data(unsigned int cpt, const std::string &video_color_images, const std::string &video_depth_images,
               bool disable_depth, const std::string &video_ground_truth,
               vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth_raw,
               unsigned int &depth_width, unsigned int &depth_height,
               std::vector<vpColVector> &pointcloud, const vpCameraParameters &cam_depth,
               vpHomogeneousMatrix &cMo_ground_truth)
{
  char buffer[FILENAME_MAX];
  // Read color
  snprintf(buffer, FILENAME_MAX, video_color_images.c_str(), cpt);
  std::string filename_color = buffer;

  if (!vpIoTools::checkFilename(filename_color)) {
    std::cerr << "Cannot read: " << filename_color << std::endl;
    return false;
  }
  vpImageIo::read(I, filename_color);

  if (!disable_depth) {
    // Read depth
    snprintf(buffer, FILENAME_MAX, video_depth_images.c_str(), cpt);
    std::string filename_depth = buffer;

    if (!vpIoTools::checkFilename(filename_depth)) {
      std::cerr << "Cannot read: " << filename_depth << std::endl;
      return false;
    }
    cv::Mat depth_raw = cv::imread(filename_depth, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (depth_raw.empty()) {
      std::cerr << "Cannot read: " << filename_depth << std::endl;
      return false;
    }

    depth_width = static_cast<unsigned int>(depth_raw.cols);
    depth_height = static_cast<unsigned int>(depth_raw.rows);
    I_depth_raw.resize(depth_height, depth_width);
    pointcloud.resize(depth_width * depth_height);

    for (int i = 0; i < depth_raw.rows; i++) {
      for (int j = 0; j < depth_raw.cols; j++) {
        I_depth_raw[i][j] = static_cast<uint16_t>(32767.5f * depth_raw.at<cv::Vec3f>(i, j)[0]);
        double x = 0.0, y = 0.0;
        // Manually limit the field of view of the depth camera
        double Z = depth_raw.at<cv::Vec3f>(i, j)[0] > 2.0f ? 0.0 : static_cast<double>(depth_raw.at<cv::Vec3f>(i, j)[0]);
        vpPixelMeterConversion::convertPoint(cam_depth, j, i, x, y);
        size_t idx = static_cast<size_t>(i * depth_raw.cols + j);
        pointcloud[idx].resize(3);
        pointcloud[idx][0] = x * Z;
        pointcloud[idx][1] = y * Z;
        pointcloud[idx][2] = Z;
      }
    }
  }

  // Read ground truth
  snprintf(buffer, FILENAME_MAX, video_ground_truth.c_str(), cpt);
  std::string filename_pose = buffer;

  cMo_ground_truth.load(filename_pose);

  return true;
}
} // namespace

void usage(const char **argv, int error, const std::string &data_path, const std::string &model_path, int first_frame)
{
  std::cout << "Synopsis" << std::endl
    << "  " << argv[0]
    << " [--data-path <path>] [--model-path <path>] [--first-frame <index>] [--depth-dense-mode <0|1>] "
    << " [--depth-normals-mode <0|1>] [--me-mode <0|1>] [--klt-mode <0|1>] [--step-by-step] [--display-ground-truth] [--help, -h]"
    << " [--P0 <stdev_P0>] [--Q <stdev_Q>] [--R <stdev_R>]"
    << std::endl
    << std::endl;
  std::cout << "Description" << std::endl
    << "  --data-path <path>   Path to the data generated by Blender get_camera_pose_teabox.py" << std::endl
    << "    Python script."
    << "    Default: " << data_path << std::endl
    << std::endl
    << "  --model-path <path>   Path to the cad model and tracker settings." << std::endl
    << "    Default: " << model_path << std::endl
    << std::endl
    << "  --first-frame <index>   First frame number to process." << std::endl
    << "    Default: " << first_frame << std::endl
    << std::endl

    << "  --depth-dense-mode  Whether to use dense depth features (0 = off, 1 = on). default: 1" << std::endl
    << std::endl
    << "  --depth-normals-mode  Whether to use normal depth features (0 = off, 1 = on). default: 0" << std::endl
    << std::endl
    << "  --me-mode  Whether to use moving edge features (0 = off, 1 = on). default: 1" << std::endl
    << std::endl
    << "  --klt-mode  Whether to use KLT features (0 = off, 1 = on). Requires OpenCV. default: 1" << std::endl
    << std::endl
    << "  --P0 Set the initial guess of the state covariance" << std::endl
    << std::endl
    << "  --Q Set the process covariance" << std::endl
    << std::endl
    << "  --R Set the measurement covariance" << std::endl
    << std::endl
    << "  --step-by-step  Flag to enable step by step mode." << std::endl
    << std::endl
    << "  --display-ground-truth  Flag to enable displaying ground truth." << std::endl
    << "    When this flag is enabled, there is no tracking. This flag is useful" << std::endl
    << "    to validate the ground truth over the rendered images." << std::endl
    << std::endl
    << "  --help, -h  Print this helper message." << std::endl
    << std::endl;
  if (error) {
    std::cout << "Error" << std::endl
      << "  "
      << "Unsupported parameter " << argv[error] << std::endl;
  }
}

int main(int argc, const char **argv)
{
  std::string opt_data_path = "data/teabox";
  std::string opt_model_path = "model/teabox";
  unsigned int opt_first_frame = 1;
  int opt_meMode = 1, opt_kltMode = 1, opt_normalsMode = 0, opt_denseMode = 1;

  bool disable_depth = false;
  bool opt_disable_klt = false;

  bool opt_display_ground_truth = false;
  bool opt_step_by_step = false;

  double opt_stdevP0 = 0.001;
  double opt_stdevQ = 0.000001;
  double opt_stdevR = 0.000001;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--data-path" && i + 1 < argc) {
      opt_data_path = std::string(argv[i + 1]);
      i++;
    }
    else if (std::string(argv[i]) == "--model-path" && i + 1 < argc) {
      opt_model_path = std::string(argv[i + 1]);
      i++;
    }
    else if (std::string(argv[i]) == "--depth-dense-mode" && i + 1 < argc) {
      opt_denseMode = static_cast<unsigned int>(atoi(argv[i + 1]));
      if (opt_denseMode < 0 || opt_denseMode > 1) {
        usage(argv, 0, opt_data_path, opt_model_path, opt_first_frame);
        return EXIT_FAILURE;
      }
      i++;
    }
    else if (std::string(argv[i]) == "--depth-normals-mode" && i + 1 < argc) {
      opt_normalsMode = static_cast<unsigned int>(atoi(argv[i + 1]));
      if (opt_normalsMode < 0 || opt_normalsMode > 1) {
        usage(argv, 0, opt_data_path, opt_model_path, opt_first_frame);
        return EXIT_FAILURE;
      }
      i++;
    }
    else if (std::string(argv[i]) == "--me-mode" && i + 1 < argc) {
      opt_meMode = static_cast<unsigned int>(atoi(argv[i + 1]));
      if (opt_meMode < 0 || opt_meMode > 1) {
        usage(argv, 0, opt_data_path, opt_model_path, opt_first_frame);
        return EXIT_FAILURE;
      }
      i++;
    }
    else if (std::string(argv[i]) == "--klt-mode" && i + 1 < argc) {
      opt_kltMode = static_cast<unsigned int>(atoi(argv[i + 1]));
      if (opt_kltMode < 0 || opt_kltMode > 1) {
        usage(argv, 0, opt_data_path, opt_model_path, opt_first_frame);
        return EXIT_FAILURE;
      }
      i++;
    }
    else if (std::string(argv[i]) == "--display-ground-truth") {
      opt_display_ground_truth = true;
    }
    else if (std::string(argv[i]) == "--step-by-step") {
      opt_step_by_step = true;
    }
    else if (std::string(argv[i]) == "--first-frame" && i + 1 < argc) {
      opt_first_frame = static_cast<unsigned int>(atoi(argv[i + 1]));
      i++;
    }
    else if (std::string(argv[i]) == "--P0" && i + 1 < argc) {
      opt_stdevP0 = std::atof(argv[i + 1]);
      ++i;
    }
    else if (std::string(argv[i]) == "--Q" && i + 1 < argc) {
      opt_stdevQ = std::atof(argv[i + 1]);
      ++i;
    }
    else if (std::string(argv[i]) == "--R" && i + 1 < argc) {
      opt_stdevR = std::atof(argv[i + 1]);
      ++i;
    }
    else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      usage(argv, 0, opt_data_path, opt_model_path, opt_first_frame);
      return EXIT_SUCCESS;
    }
    else {
      usage(argv, i, opt_data_path, opt_model_path, opt_first_frame);
      return EXIT_FAILURE;
    }
  }

  disable_depth = opt_denseMode == 0 && opt_normalsMode == 0;

  std::string video_color_images = vpIoTools::createFilePath(opt_data_path, "color/%04d_L.jpg");
  std::string video_depth_images = vpIoTools::createFilePath(opt_data_path, "depth/Image%04d_R.exr");
  std::string ground_truth = vpIoTools::createFilePath(opt_data_path, "ground-truth/Camera_L_%04d.txt");
  std::string extrinsic_file = vpIoTools::createFilePath(opt_data_path, "depth_M_color.txt");
  std::string color_camera_name = "Camera_L";
  std::string depth_camera_name = "Camera_R";
  std::string color_intrinsic_file = vpIoTools::createFilePath(opt_data_path, color_camera_name + ".xml");
  std::string depth_intrinsic_file = vpIoTools::createFilePath(opt_data_path, depth_camera_name + ".xml");
  std::string mbt_config_color = vpIoTools::createFilePath(opt_model_path, "teabox_color.xml");
  std::string mbt_config_depth = vpIoTools::createFilePath(opt_model_path, "teabox_depth.xml");
  std::string mbt_cad_model = vpIoTools::createFilePath(opt_model_path, "teabox.cao");
  std::string mbt_init_file = vpIoTools::createFilePath(opt_model_path, "teabox.init");

  std::cout << "Input data" << std::endl;
  std::cout << "  Color images    : " << video_color_images << std::endl;
  std::cout << "  Depth images    : " << (disable_depth ? "Disabled" : video_depth_images) << std::endl;
  std::cout << "  Extrinsics      : " << (disable_depth ? "Disabled" : extrinsic_file) << std::endl;
  std::cout << "  Color intrinsics: " << color_intrinsic_file << std::endl;
  std::cout << "  Depth intrinsics: " << (disable_depth ? "Disabled" : depth_intrinsic_file) << std::endl;
  std::cout << "  Ground truth    : " << ground_truth << std::endl;
  std::cout << "Tracker settings" << std::endl;
  std::cout << "  Color config    : " << mbt_config_color << std::endl;
  std::cout << "  Depth config    : " << mbt_config_depth << std::endl;
  std::cout << "  CAD model       : " << mbt_cad_model << std::endl;
  std::cout << "  First frame     : " << opt_first_frame << std::endl;
  std::cout << "  Step by step    : " << opt_step_by_step << std::endl;
  if (opt_display_ground_truth) {
    std::cout << "  Ground truth is used to project the cad model (no tracking)" << std::endl;
  }
  else {
    std::cout << "  Init file       : " << mbt_init_file << std::endl;
    std::cout << "  Features        : moving-edges " << (opt_disable_klt ? "" : "+ keypoints") << (disable_depth ? "" : " + depth") << std::endl;
  }


  std::vector<int> tracker_types;
  int colorTracker = 0;


  if (opt_meMode == 1) {
    colorTracker |= vpMbGenericTracker::EDGE_TRACKER;
  }
  if (opt_kltMode == 1) {
#if defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGPROC) && defined(HAVE_OPENCV_VIDEO)
    colorTracker |= vpMbGenericTracker::KLT_TRACKER;
#else
    std::cerr << "Warning: keypoints cannot be used as features since ViSP is not built with OpenCV 3rd party" << std::endl;
#endif
  }

  if (colorTracker == 0) {
    std::cerr << "You should use at least one type of color feature. If OpenCV is not installed, KLT features are disabled" << std::endl;
    return EXIT_FAILURE;
  }

  tracker_types.push_back(colorTracker);

  if (!disable_depth) {
    int depthTracker = 0;
    if (opt_denseMode == 1) {
      depthTracker |= vpMbGenericTracker::DEPTH_DENSE_TRACKER;
    }
    if (opt_normalsMode == 1) {
      depthTracker |= vpMbGenericTracker::DEPTH_NORMAL_TRACKER;
    }

    tracker_types.push_back(depthTracker);
  }

  vpMbGenericTracker tracker(tracker_types);
  if (!disable_depth) {
    tracker.loadConfigFile(mbt_config_color, mbt_config_depth, true);
  }
  else {
    tracker.loadConfigFile(mbt_config_color);

  }
  tracker.loadModel(mbt_cad_model);
  vpCameraParameters cam_color, cam_depth;

  // Update intrinsics camera parameters from Blender generated data
  vpXmlParserCamera p;
  if (p.parse(cam_color, color_intrinsic_file, color_camera_name, vpCameraParameters::perspectiveProjWithoutDistortion)
     != vpXmlParserCamera::SEQUENCE_OK) {
    std::cout << "Cannot found intrinsics for camera " << color_camera_name << std::endl;
  }
  if (p.parse(cam_depth, depth_intrinsic_file, depth_camera_name, vpCameraParameters::perspectiveProjWithoutDistortion)
     != vpXmlParserCamera::SEQUENCE_OK) {
    std::cout << "Cannot found intrinsics for camera " << depth_camera_name << std::endl;
  }

  if (!disable_depth)
    tracker.setCameraParameters(cam_color, cam_depth);
  else
    tracker.setCameraParameters(cam_color);

  // Reload intrinsics from tracker (useless)
  if (!disable_depth)
    tracker.getCameraParameters(cam_color, cam_depth);
  else
    tracker.getCameraParameters(cam_color);
  tracker.setDisplayFeatures(true);
  std::cout << "cam_color:\n" << cam_color << std::endl;

  if (!disable_depth)
    std::cout << "cam_depth:\n" << cam_depth << std::endl;

  vpImage<uint16_t> I_depth_raw;
  vpImage<unsigned char> I, I_depth;
  unsigned int depth_width = 0, depth_height = 0;
  std::vector<vpColVector> pointcloud;
  vpHomogeneousMatrix cMo_ground_truth;

  unsigned int frame_cpt = opt_first_frame;
  read_data(frame_cpt, video_color_images, video_depth_images, disable_depth, ground_truth,
            I, I_depth_raw, depth_width, depth_height, pointcloud, cam_depth, cMo_ground_truth);
  vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

#if defined(VISP_HAVE_X11)
  vpDisplayX d1, d2;
#elif defined(VISP_HAVE_GDI)
  vpDisplayGDI d1, d2;
#elif defined (HAVE_OPENCV_HIGHGUI)
  vpDisplayOpenCV d1, d2;
#endif

  d1.init(I, 0, 0, "Color image");
  if (!disable_depth) {
    d2.init(I_depth, static_cast<int>(I.getWidth()), 0, "Depth image");
  }

  vpHomogeneousMatrix depth_M_color;
  if (!disable_depth) {
    depth_M_color.load(extrinsic_file);
    tracker.setCameraTransformationMatrix("Camera2", depth_M_color);
    std::cout << "depth_M_color:\n" << depth_M_color << std::endl;
  }

  if (opt_display_ground_truth) {
    tracker.initFromPose(I, cMo_ground_truth); // I and I_depth must be the same size when using depth features!
  }
  else {
    tracker.initClick(I, mbt_init_file, true); // I and I_depth must be the same size when using depth features!
  }

  vpMatrix Id;
  Id.eye(6);
  vpMatrix P0 = Id * opt_stdevP0 * opt_stdevP0;
  vpMatrix Q = Id * opt_stdevQ * opt_stdevQ;
  vpMatrix R = Id * opt_stdevR * opt_stdevR;
  double alphaPred = 0.1;
  const double dt = 0.040; // 40ms <=> 25Hz
  vpUnscentedKalmanPose::State X0;
  Id.eye(3);
  vpMatrix R_ukfm = Id * opt_stdevR * opt_stdevR;
  vpUnscentedKalmanPose ukfm(Q, R_ukfm, std::vector<double>(3, alphaPred), X0, P0,
                                        vpUnscentedKalmanPose::fSE3, vpUnscentedKalmanPose::hSE3, vpUnscentedKalmanPose::phiSE3,
                                        vpUnscentedKalmanPose::phiinvSE3);
  vpHomogeneousMatrix cMo_prev, cMo_filt;
  bool isNotInitialized = true;
  try {
    bool quit = false;
    while (!quit && read_data(frame_cpt, video_color_images, video_depth_images, disable_depth,
                              ground_truth, I, I_depth_raw, depth_width, depth_height, pointcloud, cam_depth,
                              cMo_ground_truth)) {
      vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);
      vpDisplay::display(I);
      vpDisplay::display(I_depth);

      if (opt_display_ground_truth) {
        tracker.initFromPose(I, cMo_ground_truth); // I and I_depth must be the same size when using depth features!
      }
      else {
        if (!disable_depth) {
          std::map<std::string, const vpImage<unsigned char> *> mapOfImages;
          std::map<std::string, const std::vector<vpColVector> *> mapOfPointClouds;
          std::map<std::string, unsigned int> mapOfPointCloudWidths;
          std::map<std::string, unsigned int> mapOfPointCloudHeights;

          mapOfImages["Camera1"] = &I;
          mapOfPointClouds["Camera2"] = &pointcloud;
          mapOfPointCloudWidths["Camera2"] = depth_width;
          mapOfPointCloudHeights["Camera2"] = depth_height;
          tracker.track(mapOfImages, mapOfPointClouds, mapOfPointCloudWidths, mapOfPointCloudHeights);
        }
        else {
          tracker.track(I);
        }
      }

      vpHomogeneousMatrix cMo = tracker.getPose();
      vpColVector v(6, 0.);
      if (isNotInitialized) {
        ukfm.setX0(cMo);
        isNotInitialized = false;
      }
      else {
        v = vpExponentialMap::inverse(cMo_prev * cMo.inverse(), dt);
        std::cout << "v = " << v.transpose() << std::endl;
      }
      ukfm.filter(v, vpUnscentedKalmanPose::asPositionVector(cMo), dt);
      cMo_filt = ukfm.getState();

      std::cout << "\nFrame: " << frame_cpt << std::endl;
      if (!opt_display_ground_truth)
        std::cout << "cMo:\n" << cMo << std::endl;
      std::cout << "cMo_filt:\n" << cMo_filt << std::endl;
      std::cout << "cMo ground truth:\n" << cMo_ground_truth << std::endl;
      if (!disable_depth) {
        tracker.display(I, I_depth, cMo, depth_M_color * cMo, cam_color, cam_depth, vpColor::red, 2);
        vpDisplay::displayFrame(I_depth, depth_M_color * cMo, cam_depth, 0.05, vpColor::none, 2);
      }
      else {
        tracker.display(I, cMo, cam_color, vpColor::red, 2);
      }

      vpDisplay::displayFrame(I, cMo, cam_color, 0.05, vpColor::none, 2);
      vpDisplay::displayFrame(I, cMo_filt, cam_color, 0.05, vpColor::yellow, 3);
      std::ostringstream oss;
      oss << "Frame: " << frame_cpt;
      vpDisplay::setTitle(I, oss.str());
      if (opt_step_by_step) {
        vpDisplay::displayText(I, 20, 10, "Left click to trigger next step", vpColor::red);
        vpDisplay::displayText(I, 40, 10, "Right click to quit step-by-step mode", vpColor::red);
      }
      else {
        vpDisplay::displayText(I, 20, 10, "Left click to trigger step-by-step mode", vpColor::red);
        vpDisplay::displayText(I, 40, 10, "Right click to exit...", vpColor::red);
      }
      if (!opt_display_ground_truth) {
        {
          std::stringstream ss;
          ss << "Nb features: " << tracker.getError().size();
          vpDisplay::displayText(I, I.getHeight() - 50, 20, ss.str(), vpColor::red);
        }
        {
          std::stringstream ss;
          ss << "Features: edges " << tracker.getNbFeaturesEdge() << ", klt " << tracker.getNbFeaturesKlt()
            << ", depth " << tracker.getNbFeaturesDepthDense();
          vpDisplay::displayText(I, I.getHeight() - 30, 20, ss.str(), vpColor::red);
        }
      }

      vpDisplay::flush(I);
      vpDisplay::flush(I_depth);

      // Button 1: start step by step if not enabled from command line option
      // Button 2: enables step by step mode
      // Button 3: ends step by step mode if enabled
      //           quit otherwise
      vpMouseButton::vpMouseButtonType button;
      if (vpDisplay::getClick(I, button, opt_step_by_step)) {
        if (button == vpMouseButton::button1 && opt_step_by_step == false) {
          opt_step_by_step = true;
        }
        else if (button == vpMouseButton::button3 && opt_step_by_step == true) {
          opt_step_by_step = false;
        }
        else if (button == vpMouseButton::button3 && opt_step_by_step == false) {
          quit = true;
        }
        else if (button == vpMouseButton::button2) {
          opt_step_by_step = true;
        }
      }

      frame_cpt++;
      cMo_prev = cMo;
    }

    vpDisplay::flush(I);
    vpDisplay::getClick(I);
  }
  catch (std::exception &e) {
    std::cerr << "Catch exception: " << e.what() << std::endl;
  }

  return EXIT_SUCCESS;
}
#else
int main()
{
  std::cout << "To run this tutorial, ViSP should be built with OpenCV and pugixml libraries." << std::endl;
  return EXIT_SUCCESS;
}
#endif
