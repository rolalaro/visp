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
 *
 * Description:
 * Display a point cloud using PCL library.
 */

/*!
  \file vpUnscentedKalmanPose.cpp
  \brief Unscented Kalman for pose filtering on SE(3).
*/

#include <visp3/core/vpUnscentedKalmanPose.h>

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
BEGIN_VISP_NAMESPACE
vpUnscentedKalmanPose::vpUnscentedKalmanPose(const vpMatrix &Q, const vpMatrix &R, const double &alphaPred, const double &alphaUpdate)
  : m_alphaPredict(alphaPred)
  , m_Q(Q)
  , m_alphaUpdate(alphaUpdate)
  , m_R(R)

{ }

void vpUnscentedKalmanPose::init(const vpHomogeneousMatrix &X0, const vpMatrix &P0, const vpColVector &omega0)
{
  if ((m_Q.getCols() != P0.getCols()) || (m_Q.getRows() != P0.getRows())) {
    throw(vpException(vpException::dimensionError, "Initial process covariance matrix P0 and Q matrix sizes mismatch"));
  }
  m_Xpred = X0;
  m_Ppred = P0;
  m_Xest = X0;
  m_Pest = P0;
  m_omega = omega0;
}

void vpUnscentedKalmanPose::filter(const vpColVector &z, const double &dt)
{
  predict(dt);
  update(z, dt);
}

void vpUnscentedKalmanPose::predict(const double &dt)
{
  // Drawing the sigma points and associated weights
  vpSigmaPointDrawingResult sigmaPoints = sigmaPointsDrawingPredict();

  // Computation of the mean and covariance of the prior
  unscentedTransformPredict(m_Xest, sigmaPoints.m_chis, sigmaPoints.m_wc, m_omega, dt);
}

void vpUnscentedKalmanPose::update(const vpColVector &z, const double &dt)
{
  // Drawing of the sigma points for the update step
  vpSigmaPointDrawingResult sigmaPoints = sigmaPointsDrawingUpdate();

  // Computation of the mean and covariance of the prior expressed in the measurement space
  vpUnscentedTransformResult transformResults = unscentedTransformUpdate(sigmaPoints.m_chis, sigmaPoints.m_wm, sigmaPoints.m_wc, dt);
  m_Pz = transformResults.m_P;

  // Computation of cross covariance
  m_Pxz = sigmaPoints.m_wc[0] *(sigmaPoints.m_chis[0] - sigmaPoints.m_muchis) * (transformResults.m_z[0] - transformResults.m_mu).transpose();
  size_t nbPts = sigmaPoints.m_wc.size();
  for (size_t i = 1; i < nbPts; ++i) {
    m_Pxz += sigmaPoints.m_wc[i] *(sigmaPoints.m_chis[i] - sigmaPoints.m_muchis) * (transformResults.m_z[i] - transformResults.m_mu).transpose();
  }

  // Computation of the Kalman gain
  m_K = m_Pxz * m_Pz.inverseByCholesky();

  // Updating the estimate
  vpColVector temp = m_K * (z - transformResults.m_mu);
  vpColVector epsilon = temp.extract(0, m_q);
  m_omega = epsilon;
  m_Xest = m_Xpred * vpExponentialMap::direct(epsilon);
  m_Pest = m_Ppred - m_K * m_Pz * m_K.transpose();
}

vpUnscentedKalmanPose::vpSigmaPointDrawingResult vpUnscentedKalmanPose::sigmaPointsDrawingPredict()
{
  const double lambda = ((m_alphaPredict * m_alphaPredict)- 1.) * 2. * static_cast<double>(m_q);
  const unsigned int nbSigmaPoints = 4 * m_q;
  const unsigned int halfNbSigmaPoints = nbSigmaPoints / 2;
  const double commonWeight = 0.5/(lambda + 2 * m_q);

  vpSigmaPointDrawingResult results;
  results.m_chis.resize(nbSigmaPoints);
  results.m_wc.resize(nbSigmaPoints);

  vpMatrix Paug(2 * m_q, 2 * m_q, 0.);
  Paug.insert(m_Pest, 0, 0);
  Paug.insert(m_Q, m_q, m_q);
  vpMatrix squareRootPaug = ((2. * static_cast<double>(m_q) + lambda) * Paug).cholesky();

  for (unsigned int i = 0; i < halfNbSigmaPoints; ++i) {
    results.m_wc[2*i] = commonWeight;
    results.m_wc[2*i + 1] = commonWeight;
    results.m_chis[i] = squareRootPaug.getCol(i);
    results.m_chis[halfNbSigmaPoints + i] = -1. * squareRootPaug.getCol(i);
  }
  return results;
}

void vpUnscentedKalmanPose::unscentedTransformPredict(const vpHomogeneousMatrix &Xprev, const std::vector<vpColVector> &sigmaPoints,
    const std::vector<double> &wc, const vpColVector &omega, const double &dt)
{
  // Computation of the mean
  m_Xpred = Xprev * vpExponentialMap::direct(omega, dt);

  // Computation of the covariance
  m_Ppred.resize(6, 6, 0.);
  size_t nbSigmaPoints = sigmaPoints.size();
  for (size_t i = 0; i < nbSigmaPoints; ++i) {
    vpColVector epsilonj = sigmaPoints[i].extract(0, 6);
    vpColVector wj = sigmaPoints[i].extract(6, 6);
    vpHomogeneousMatrix epsilon = vpExponentialMap::direct(omega, dt).inverse() * vpExponentialMap::direct(epsilonj, dt) * vpExponentialMap::direct(omega + wj);
    vpColVector logEpsilon = vpExponentialMap::inverse(epsilon, dt);
    m_Ppred += wc[i] * logEpsilon * logEpsilon.transpose();
  }
}

vpUnscentedKalmanPose::vpSigmaPointDrawingResult vpUnscentedKalmanPose::sigmaPointsDrawingUpdate()
{
  const unsigned int l = m_q + m_k;
  const double lambda = ((m_alphaUpdate * m_alphaUpdate)- 1.) * static_cast<double>(l);
  const unsigned int nbSigmaPoints = 2 * l;
  const double commonWeight = 0.5/(lambda + static_cast<double>(l));

  vpSigmaPointDrawingResult results;
  results.m_chis.resize(nbSigmaPoints);
  results.m_wm.resize(nbSigmaPoints);
  results.m_wc.resize(nbSigmaPoints);
  results.m_wm[0] = lambda / (lambda + static_cast<double>(l));
  results.m_wc[0] = (lambda / (lambda + static_cast<double>(l))) + (3. - m_alphaUpdate * m_alphaUpdate);
  results.m_muchis = vpColVector(l, 0.);
  results.m_muchis.insert(m_q, m_muNoiseMeas);

  vpMatrix Paug(l, l, 0.);
  Paug.insert(m_Ppred, 0, 0);
  Paug.insert(m_R, m_q, m_q);
  vpMatrix squareRootPaug = ((static_cast<double>(l) + lambda) * Paug).cholesky();

  for (unsigned int i = 0; i < nbSigmaPoints; ++i) {
    results.m_wm[i] = commonWeight;
    results.m_wc[i] = commonWeight;
    if (i < l) {
      results.m_chis[i] = results.m_muchis + squareRootPaug.getCol(i);
    }
    else {
      results.m_chis[i] = results.m_muchis - squareRootPaug.getCol(i - l);
    }
  }

  return results;
}

vpUnscentedKalmanPose::vpUnscentedTransformResult vpUnscentedKalmanPose::unscentedTransformUpdate(
  const std::vector<vpColVector> &sigmaPoints,
  const std::vector<double> &wm, const std::vector<double> &wc,
  const double &dt
)
{
  const unsigned int nbSigmaPoints = sigmaPoints.size();
  vpUnscentedKalmanPose::vpUnscentedTransformResult result;
  result.m_mu = vpColVector(m_q, 0.);
  result.m_P = vpMatrix(m_q, m_q, 0.);
  // Computation of the mean of the chi points projected in the measurement space
  for (unsigned int i = 0; i < nbSigmaPoints; ++i) {
    vpColVector epsilon = sigmaPoints[i].extract(0, m_q);
    vpColVector v = sigmaPoints[i].extract(m_q, m_k);
    vpHomogeneousMatrix temp = vpExponentialMap::direct(epsilon, dt) * vpExponentialMap::direct(v, dt);
    vpColVector z = vpExponentialMap::inverse(temp, dt);
    result.m_z.push_back(z);
    result.m_mu += wm[i] * z;
  }

  // Computation of the state covariance P_{zz}
  for (unsigned int i = 0; i < nbSigmaPoints; ++i) {
    vpColVector diff = result.m_z[i] - result.m_mu;
    result.m_P += wc[i] * diff * diff.transpose();
  }

  return result;
}
END_VISP_NAMESPACE
#else
void vpUnscentedKalman_dummy()
{

}
#endif
