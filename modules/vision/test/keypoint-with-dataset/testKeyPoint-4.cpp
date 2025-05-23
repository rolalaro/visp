/*
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2025 by Inria. All rights reserved.
 *
 * This software is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * See the file LICENSE.txt at the root directory of this source
 * distribution for additional information about the GNU GPL.
 *
 * For using ViSP with software that can not be combined with the GNU
 * GPL, please contact Inria about acquiring a ViSP Professional
 * Edition License.
 *
 * See https://visp.inria.fr for more information.
 *
 * This software was developed at:
 * Inria Rennes - Bretagne Atlantique
 * Campus Universitaire de Beaulieu
 * 35042 Rennes Cedex
 * France
 *
 * If you have questions regarding the use of this file, please contact
 * Inria at visp@inria.fr
 *
 * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 * WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Description:
 * Test keypoint matching and pose estimation with mostly OpenCV functions
 * calls to detect potential memory leaks in testKeyPoint-2.cpp.
 */

/*!
  \example testKeyPoint-4.cpp

  \brief   Test keypoint matching and pose estimation with mostly OpenCV
  functions calls to detect potential memory leaks in testKeyPoint-2.cpp.
*/

#include <iostream>

#include <visp3/core/vpConfig.h>

#if defined(VISP_HAVE_OPENCV) && defined(HAVE_OPENCV_IMGPROC) && \
  (((VISP_HAVE_OPENCV_VERSION < 0x050000)  && defined(HAVE_OPENCV_CALIB3D) && defined(HAVE_OPENCV_FEATURES2D)) || \
   ((VISP_HAVE_OPENCV_VERSION >= 0x050000) && defined(HAVE_OPENCV_3D) && defined(HAVE_OPENCV_FEATURES)))

#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/io/vpParseArgv.h>
#include <visp3/io/vpVideoReader.h>
#include <visp3/mbt/vpMbEdgeTracker.h>
#include <visp3/vision/vpKeyPoint.h>

// List of allowed command line options
#define GETOPTARGS "cdh"

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

void usage(const char *name, const char *badparam);
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display);

/*!

  Print the program options.

  \param name : Program name.
  \param badparam : Bad parameter name.

*/
void usage(const char *name, const char *badparam)
{
  fprintf(stdout, "\n\
Test keypoints matching.\n\
\n\
SYNOPSIS\n\
  %s [-c] [-d] [-h]\n",
    name);

  fprintf(stdout, "\n\
OPTIONS:                                               \n\
\n\
  -c\n\
     Disable the mouse click. Useful to automate the \n\
     execution of this program without human intervention.\n\
\n\
  -d \n\
     Turn off the display.\n\
\n\
  -h\n\
     Print the help.\n");

  if (badparam)
    fprintf(stdout, "\nERROR: Bad parameter [%s]\n", badparam);
}

/*!

  Set the program options.

  \param argc : Command line number of parameters.
  \param argv : Array of command line parameters.
  \param click_allowed : Mouse click activation.
  \param display : Display activation.
  \return false if the program has to be stopped, true otherwise.

*/
bool getOptions(int argc, const char **argv, bool &click_allowed, bool &display)
{
  const char *optarg_;
  int c;
  while ((c = vpParseArgv::parse(argc, argv, GETOPTARGS, &optarg_)) > 1) {

    switch (c) {
    case 'c':
      click_allowed = false;
      break;
    case 'd':
      display = false;
      break;
    case 'h':
      usage(argv[0], nullptr);
      return false;

    default:
      usage(argv[0], optarg_);
      return false;
    }
  }

  if ((c == 1) || (c == -1)) {
    // standalone param or error
    usage(argv[0], nullptr);
    std::cerr << "ERROR: " << std::endl;
    std::cerr << "  Bad argument " << optarg_ << std::endl << std::endl;
    return false;
  }

  return true;
}

template <typename Type>
void run_test(const std::string &env_ipath, bool opt_click_allowed, bool opt_display, vpImage<Type> &I,
              vpImage<Type> &Imatch, vpImage<Type> &Iref)
{
#if defined(VISP_HAVE_DATASET)
#if VISP_HAVE_DATASET_VERSION >= 0x030600
  std::string ext("png");
#else
  std::string ext("pgm");
#endif
#else
  // We suppose that the user will download a recent dataset
  std::string ext("png");
#endif
  // Set the path location of the image sequence
  std::string dirname = vpIoTools::createFilePath(env_ipath, "mbt/cube");

  // Build the name of the image files
  std::string filenameRef = vpIoTools::createFilePath(dirname, "image0000." + ext);
  vpImageIo::read(I, filenameRef);
  Iref = I;
  std::string filenameCur = vpIoTools::createFilePath(dirname, "image%04d." + ext);

  vpDisplay *display = nullptr, *display2 = nullptr;

  if (opt_display) {
    Imatch.resize(I.getHeight(), 2 * I.getWidth());
    Imatch.insert(I, vpImagePoint(0, 0));

#ifdef VISP_HAVE_DISPLAY
    display = vpDisplayFactory::allocateDisplay(I, 0, 0, "ORB keypoints matching");
    display->setDownScalingFactor(vpDisplay::SCALE_AUTO);
    display2 = vpDisplayFactory::allocateDisplay(Imatch, 0, static_cast<int>(I.getHeight()) / vpDisplay::getDownScalingFactor(I) + 40, "ORB keypoints matching");
    display2->setDownScalingFactor(vpDisplay::SCALE_AUTO);
#else
    std::cout << "No image viewer is available..." << std::endl;
#endif
  }

  vpCameraParameters cam;
  vpMbEdgeTracker tracker;
  // Load config for tracker
  std::string tracker_config_file = vpIoTools::createFilePath(env_ipath, "mbt/cube.xml");

#if defined(VISP_HAVE_PUGIXML)
  tracker.loadConfigFile(tracker_config_file);
  tracker.getCameraParameters(cam);
#else
  // Corresponding parameters manually set to have an example code
  vpMe me;
  me.setMaskSize(5);
  me.setMaskNumber(180);
  me.setRange(8);
  me.setLikelihoodThresholdType(vpMe::NORMALIZED_THRESHOLD);
  me.setThreshold(20);
  me.setMu1(0.5);
  me.setMu2(0.5);
  me.setSampleStep(4);
  me.setNbTotalSample(250);
  tracker.setMovingEdge(me);
  cam.initPersProjWithoutDistortion(547.7367575, 542.0744058, 338.7036994, 234.5083345);
  tracker.setCameraParameters(cam);
  tracker.setNearClippingDistance(0.01);
  tracker.setFarClippingDistance(100.0);
  tracker.setClipping(tracker.getClipping() | vpMbtPolygon::FOV_CLIPPING);
#endif

  tracker.setAngleAppear(vpMath::rad(89));
  tracker.setAngleDisappear(vpMath::rad(89));

  // Load CAO model
  std::string cao_model_file = vpIoTools::createFilePath(env_ipath, "mbt/cube.cao");
  tracker.loadModel(cao_model_file);

  // Initialize the pose
  std::string init_file = vpIoTools::createFilePath(env_ipath, "mbt/cube.init");
  if (opt_display && opt_click_allowed) {
    tracker.initClick(I, init_file);
  }
  else {
    vpHomogeneousMatrix cMoi(0.02044769891, 0.1101505452, 0.5078963719, 2.063603907, 1.110231561, -0.4392789872);
    tracker.initFromPose(I, cMoi);
  }

  // Get the init pose
  vpHomogeneousMatrix cMo;
  tracker.getPose(cMo);

  // Init keypoints
  cv::Ptr<cv::FeatureDetector> detector;
  cv::Ptr<cv::DescriptorExtractor> extractor;
  cv::Ptr<cv::DescriptorMatcher> matcher;

#if defined(VISP_HAVE_OPENCV) && \
    (((VISP_HAVE_OPENCV_VERSION < 0x050000) && defined(HAVE_OPENCV_FEATURES2D)) || \
     ((VISP_HAVE_OPENCV_VERSION >= 0x050000) && defined(HAVE_OPENCV_FEATURES)))
#if (VISP_HAVE_OPENCV_VERSION >= 0x030000)
  detector = cv::ORB::create(500, 1.2f, 1);
  extractor = cv::ORB::create(500, 1.2f, 1);
#elif (VISP_HAVE_OPENCV_VERSION >= 0x020301)
  detector = cv::FeatureDetector::create("ORB");
  extractor = cv::DescriptorExtractor::create("ORB");
#endif
#endif
  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  // Detect keypoints on the current image
  std::vector<cv::KeyPoint> trainKeyPoints;
  cv::Mat matImg;
  vpImageConvert::convert(I, matImg);
  detector->detect(matImg, trainKeyPoints);

  // Keep only keypoints on the cube
  std::vector<vpPolygon> polygons;
  std::vector<std::vector<vpPoint> > roisPt;
  std::pair<std::vector<vpPolygon>, std::vector<std::vector<vpPoint> > > pair = tracker.getPolygonFaces(false);
  polygons = pair.first;
  roisPt = pair.second;

  // Compute the 3D coordinates
  std::vector<cv::Point3f> points3f;
  vpKeyPoint::compute3DForPointsInPolygons(cMo, cam, trainKeyPoints, polygons, roisPt, points3f);

  // Extract descriptors
  cv::Mat trainDescriptors;
  extractor->compute(matImg, trainKeyPoints, trainDescriptors);

  if (trainKeyPoints.size() != static_cast<size_t>(trainDescriptors.rows) || trainKeyPoints.size() != points3f.size()) {
    throw(vpException(vpException::fatalError, "Problem with training data size !"));
  }

  // Init reader for getting the input image sequence
  vpVideoReader g;
  g.setFileName(filenameCur);
  g.open(I);
  g.acquire(I);

  bool opt_click = false;
  vpMouseButton::vpMouseButtonType button;
  while (g.getFrameIndex() < 30) {
    g.acquire(I);

    vpImageConvert::convert(I, matImg);
    std::vector<cv::KeyPoint> queryKeyPoints;
    detector->detect(matImg, queryKeyPoints);

    cv::Mat queryDescriptors;
    extractor->compute(matImg, queryKeyPoints, queryDescriptors);

    std::vector<std::vector<cv::DMatch> > knn_matches;
    std::vector<cv::DMatch> matches;
    matcher->knnMatch(queryDescriptors, trainDescriptors, knn_matches, 2);
    for (std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end();
      ++it) {
      if (it->size() > 1) {
        double ratio = (*it)[0].distance / (*it)[1].distance;
        if (ratio < 0.85) {
          matches.push_back((*it)[0]);
        }
      }
    }

    vpPose estimated_pose;
    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
      vpPoint pt(points3f[static_cast<size_t>(it->trainIdx)].x, points3f[static_cast<size_t>(it->trainIdx)].y,
        points3f[static_cast<size_t>(it->trainIdx)].z);

      double x = 0.0, y = 0.0;
      vpPixelMeterConversion::convertPoint(cam, queryKeyPoints[static_cast<size_t>(it->queryIdx)].pt.x,
        queryKeyPoints[static_cast<size_t>(it->queryIdx)].pt.y, x, y);
      pt.set_x(x);
      pt.set_y(y);

      estimated_pose.addPoint(pt);
    }

    bool is_pose_estimated = false;
    if (estimated_pose.npt >= 4) {
      try {
        unsigned int nb_inliers = static_cast<unsigned int>(0.7 * estimated_pose.npt);
        estimated_pose.setRansacNbInliersToReachConsensus(nb_inliers);
        estimated_pose.setRansacThreshold(0.001);
        estimated_pose.setRansacMaxTrials(500);
        if (estimated_pose.computePose(vpPose::RANSAC, cMo)) {
          is_pose_estimated = true; // success
        }
        else {
          is_pose_estimated = false;
        }
      }
      catch (...) {
        is_pose_estimated = false;
      }
    }

    if (opt_display) {
      vpDisplay::display(I);

      Imatch.insert(I, vpImagePoint(0, Iref.getWidth()));
      vpDisplay::display(Imatch);
      for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        vpImagePoint leftPt(trainKeyPoints[static_cast<size_t>(it->trainIdx)].pt.y, trainKeyPoints[static_cast<size_t>(it->trainIdx)].pt.x);
        vpImagePoint rightPt(queryKeyPoints[static_cast<size_t>(it->queryIdx)].pt.y,
          queryKeyPoints[static_cast<size_t>(it->queryIdx)].pt.x + Iref.getWidth());
        vpDisplay::displayLine(Imatch, leftPt, rightPt, vpColor::green);
      }

      if (is_pose_estimated) {
        tracker.setPose(I, cMo);
        tracker.display(I, cMo, cam, vpColor::red);
        vpDisplay::displayFrame(I, cMo, cam, 0.05, vpColor::none);
      }

      vpDisplay::flush(Imatch);
      vpDisplay::flush(I);
    }

    // Click requested to process next image
    if (opt_click_allowed && opt_display) {
      if (opt_click) {
        vpDisplay::getClick(I, button, true);
        if (button == vpMouseButton::button3) {
          opt_click = false;
        }
      }
      else {
        // Use right click to enable/disable step by step tracking
        if (vpDisplay::getClick(I, button, false)) {
          if (button == vpMouseButton::button3) {
            opt_click = true;
          }
          else if (button == vpMouseButton::button1) {
            break;
          }
        }
      }
    }
  }

  if (display) {
    delete display;
  }
  if (display2) {
    delete display2;
  }
}

int main(int argc, const char **argv)
{
  try {
    std::string env_ipath;
    bool opt_click_allowed = true;
    bool opt_display = true;

    // Read the command line options
    if (getOptions(argc, argv, opt_click_allowed, opt_display) == false) {
      return EXIT_FAILURE;
    }

    // Get the visp-images-data package path or VISP_INPUT_IMAGE_PATH
    // environment variable value
    env_ipath = vpIoTools::getViSPImagesDataPath();

    if (env_ipath.empty()) {
      std::cerr << "Please set the VISP_INPUT_IMAGE_PATH environment "
        "variable value."
        << std::endl;
      return EXIT_FAILURE;
    }

    {
      vpImage<unsigned char> I, Imatch, Iref;

      std::cout << "-- Test on gray level images" << std::endl;
      run_test(env_ipath, opt_click_allowed, opt_display, I, Imatch, Iref);
    }

    {
      vpImage<vpRGBa> I, Imatch, Iref;

      std::cout << "-- Test on color images" << std::endl;
      run_test(env_ipath, opt_click_allowed, opt_display, I, Imatch, Iref);
    }

  }
  catch (const vpException &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "testKeyPoint-4 is ok !" << std::endl;
  return EXIT_SUCCESS;
}

#else
int main()
{
  std::cerr << "You need OpenCV library." << std::endl;

  return EXIT_SUCCESS;
}

#endif
