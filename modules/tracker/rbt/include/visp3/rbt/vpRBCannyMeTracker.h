/*
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2024 by Inria. All rights reserved.
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
 */

/*!
  \file vpRBSilhouetteMeTracker.h
  \brief Moving edge tracking for depth-extracted object contours
*/
#ifndef VP_RB_CANNY_ME_TRACKER_H
#define VP_RB_CANNY_ME_TRACKER_H

#include <visp3/rbt/vpRBFeatureTracker.h>
#include <visp3/rbt/vpRBBaseMeTracker.h>
#include <visp3/rbt/vpRBSilhouetteControlPoint.h>
#include <visp3/core/vpCannyEdgeDetection.h>
#include <visp3/core/vpRobust.h>

BEGIN_VISP_NAMESPACE
/**
 * \brief Moving edge feature tracking from depth-extracted object contours
 *
 * \ingroup group_rbt_trackers
*/
class VISP_EXPORT vpRBCannyMeTracker : public vpRBBaseMeTracker
{
public:

  vpRBCannyMeTracker() :
    vpRBBaseMeTracker(), m_cannyDetector()
  {
    init();
  }

  inline void init()
  {
    m_cannyDetector.setCannyThresholds(-1., -1.); // Force to use automatic thresholding
    m_cannyDetector.setStoreEdgePoints(true); // Force to store the edge points list
  }

  virtual ~vpRBCannyMeTracker() = default;

  bool requiresRGB() const VP_OVERRIDE { return false; }

  bool requiresDepth() const VP_OVERRIDE { return false; }

  bool requiresSilhouetteCandidates() const VP_OVERRIDE { return true; }

  void onTrackingIterEnd(const vpHomogeneousMatrix & /*cMo*/) VP_OVERRIDE { }

  /**
   * @brief Extract the geometric features from the list of collected silhouette points
   */
  void extractFeatures(const vpRBFeatureTrackerInput &frame, const vpRBFeatureTrackerInput &previousFrame, const vpHomogeneousMatrix &cMo) VP_OVERRIDE;

  void display(const vpCameraParameters &cam, const vpImage<unsigned char> &I, const vpImage<vpRGBa> &IRGB, const vpImage<unsigned char> &depth) const VP_OVERRIDE;

#if defined(VISP_HAVE_NLOHMANN_JSON)
  virtual void loadJsonConfiguration(const nlohmann::json &j) VP_OVERRIDE
  {
    vpRBBaseMeTracker::loadJsonConfiguration(j);
    m_cannyDetector = j.at("canny");
    init();
  }
#endif

  /**
   * \name Settings
   * @}
   */

   /**
   * \see vpCannyEdgeDetection::setFilteringAndGradientType()
   */
  inline void setFilteringAndGradientType(const vpImageFilter::vpCannyFilteringAndGradientType &type)
  {
    m_cannyDetector.setFilteringAndGradientType(type);
  }

  /**
   * \see vpCannyEdgeDetection::setGradients()
   */
  inline void setGradients(const vpImage<float> &dIx, const vpImage<float> &dIy)
  {
    m_cannyDetector.setGradients(dIx, dIy);
  }

  /**
   * \see vpCannyEdgeDetection::setCannyThresholdsRatio()
   */
  inline void setCannyThresholdsRatio(const float &lowerThreshRatio, const float &upperThreshRatio)
  {
    m_cannyDetector.setCannyThresholdsRatio(lowerThreshRatio, upperThreshRatio);
  }

  /**
   * \see vpCannyEdgeDetection::setGaussianFilterParameters()
   */
  inline void setGaussianFilterParameters(const int &kernelSize, const float &stdev)
  {
    m_cannyDetector.setGaussianFilterParameters(kernelSize, stdev);
  }

  /**
   * \see vpCannyEdgeDetection::setGradientFilterAperture()
   */
  inline void setGradientFilterAperture(const unsigned int &apertureSize)
  {
    m_cannyDetector.setGradientFilterAperture(apertureSize);
  }

  /**
   * \see vpCannyEdgeDetection::setMask()
   */
  inline void setMask(const vpImage<bool> *p_mask)
  {
    m_cannyDetector.setMask(p_mask);
  }

  /**
   * \see vpCannyEdgeDetection::setMinimumStackSize()
   */
  inline void setMinimumStackSize(const rlim_t &requiredStackSize)
  {
    m_cannyDetector.setMinimumStackSize(requiredStackSize);
  }

  /**
   * \see vpCannyEdgeDetection::setNbThread()
   */
  inline void setNbThread(const int &maxNbThread)
  {
    m_cannyDetector.setNbThread(maxNbThread);
  }

  /**
   * \name Settings
   * @}
   */
  const vpCannyEdgeDetection &getCannyDetector() const
  {
    return m_cannyDetector;
  }

private:
  vpCannyEdgeDetection m_cannyDetector;
};

END_VISP_NAMESPACE

#endif
