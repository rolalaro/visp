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

#include <visp3/rbt/vpRBCannyMeTracker.h>

#include <array>
#define VISP_DEBUG_ME_TRACKER 0

BEGIN_VISP_NAMESPACE
namespace
{
float
getGradientOrientation(const vpImage<float> &dIx, const vpImage<float> &dIy, const int &iter)
{
  float gradientOrientation = 0.f;
  float dx = dIx.bitmap[iter];
  float dy = dIy.bitmap[iter];

  if (std::abs(dx) < std::numeric_limits<float>::epsilon()) {
    gradientOrientation = M_PI_2_FLOAT;
  }
  else {
    gradientOrientation = static_cast<float>(std::atan2(dy, dx));
  }
  return gradientOrientation;
}
}

/**
 * @brief Extract the geometric features from the list of collected silhouette points
*/
void vpRBCannyMeTracker::extractFeatures(const vpRBFeatureTrackerInput &frame, const vpRBFeatureTrackerInput &previousFrame, const vpHomogeneousMatrix &/*cMo*/)
{
  m_controlPoints.clear();
  m_controlPoints.reserve(frame.silhouettePoints.size());
  const vpHomogeneousMatrix &cMo = frame.renders.cMo;
  const vpHomogeneousMatrix oMc = cMo.inverse();
  const vpColVector oC = oMc.getRotationMatrix() * vpColVector({ 0.0, 0.0, -1.0 });
  const vpImage<unsigned char> &initImage = previousFrame.I.getSize() == frame.I.getSize() ? previousFrame.I : frame.I;

  // generate mask for edge dectector
  vpRect bbox = frame.renders.boundingBox;
  unsigned int rstart = bbox.getTop();
  unsigned int rend = bbox.getBottom();
  unsigned int left = bbox.getLeft();
  unsigned int bboxWidth = bbox.getWidth();
  unsigned int wImage = initImage.getCols();
  unsigned int nbBites = bboxWidth * sizeof(bool);
  vpImage<bool> mask(initImage.getRows(), wImage, false);
  vpImage<bool> rowMask(1, bboxWidth, true);

  for (unsigned int r = rstart; r <= rend; ++r) {
    std::memcpy(mask.bitmap + r * wImage + left, rowMask.bitmap, nbBites);
  }
  m_cannyDetector.setMask(&mask);

  // Compute the edge map
  vpImage<unsigned char> Icanny = m_cannyDetector.detect(initImage);

  // checkEdgeList(m_cannyDetector, Icanny);

  const std::vector<vpImagePoint> &edgeList = m_cannyDetector.getEdgePointsList();
  const vpImage<float> &dIx = m_cannyDetector.getGIx();
  const vpImage<float> &dIy = m_cannyDetector.getGIy();

#ifdef VISP_HAVE_OPENMP
#pragma omp parallel
#endif
  {
    std::vector<vpRBSilhouetteControlPoint> localPoints;
#ifdef VISP_HAVE_OPENMP
#pragma omp for nowait
#endif
    for (const vpImagePoint &ep: edgeList) {
#if VISP_DEBUG_ME_TRACKER
      if (sp.Z == 0) {
        throw vpException(vpException::badValue, "Got a point with Z == 0");
      }
      if (std::isnan(sp.orientation)) {
        throw vpException(vpException::badValue, "Got a point with theta nan");
      }
#endif
      vpRBSilhouetteControlPoint p;
      int i = static_cast<int>(ep.get_i());
      int j = static_cast<int>(ep.get_j());

      if (frame.renders.isSilhouette[i][j]) {
        continue;
      }

      float Z = frame.renders.depth[i][j];
      if (Z <= 0.) {
        continue;
      }

      int iter = i * wImage + j;
      vpColVector normal(3);
      normal[0] = frame.renders.normals[i][j].R;
      normal[1] = frame.renders.normals[i][j].G;
      normal[2] = frame.renders.normals[i][j].B;
      normal.normalize();
      float orientation = getGradientOrientation(dIx, dIy, iter);

      p.buildPoint(i, j, frame.renders.depth[i][j], orientation, normal, cMo, oMc, frame.cam, m_me, false);
      if (p.tooCloseToBorder(frame.I.getHeight(), frame.I.getWidth(), m_me.getRange())) {
        continue;
      }
      if (m_useMask && frame.hasMask()) {
        double maxMaskGradient;
        if (p.isSilhouette()) { // If it is a silhouette point, we check that the mask actually considers it an object border
          maxMaskGradient = p.getMaxMaskGradientAlongLine(frame.mask, m_me.getRange());
        }
        else { // Otherwise, we just check that the site is considered as belonging to the object
          maxMaskGradient = frame.mask[i][j];
        }
        if (maxMaskGradient < m_minMaskConfidence) {
          continue;
        }
      }

      p.initControlPoint(initImage, 0);
      p.setNumCandidates(m_numCandidates);
      localPoints.push_back(std::move(p));
    }

#ifdef VISP_HAVE_OPENMP
#pragma omp critical
#endif
    {
      m_controlPoints.insert(m_controlPoints.end(), localPoints.begin(), localPoints.end());
    }
  }
  m_numFeatures = m_controlPoints.size();

  m_robust.setMinMedianAbsoluteDeviation(m_robustMadMin / frame.cam.get_px());
}

void
vpRBCannyMeTracker::display(const vpCameraParameters &cam, const vpImage<unsigned char> &I, const vpImage<vpRGBa> &IRGB, const vpImage<unsigned char> &depth)  const
{
  vpRBBaseMeTracker::display(cam, I, IRGB, depth);
  const std::vector<vpImagePoint> &edgeList = m_cannyDetector.getEdgePointsList();
  for (const vpImagePoint &ep: edgeList) {
    vpDisplay::displayPoint(IRGB, ep, vpColor::yellow);
  }
  for (const vpRBSilhouetteControlPoint &ep: m_controlPoints) {
    vpDisplay::displayPoint(IRGB, ep.icpoint, vpColor::blue);
  }
}

END_VISP_NAMESPACE
