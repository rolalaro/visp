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
const double  vpUnscentedKalmanPose::TOL = 1e-9;

vpUnscentedKalmanPose::vpUnscentedKalmanPose(const vpMatrix &Q, const vpMatrix &R, const std::vector<double> &alphas,
      const State &X0, const vpMatrix &P0,
      const ProcessFunction &f, const ObservationFunction &h,
      const RetractationFunction &phi, const InverseRetractationFunction &phi_inv)
  : m_f(f)
  , m_h(h)
  , m_phi(phi)
  , m_phiinv(phi_inv)
  , m_Q(Q)
  , m_R(R)
  , m_P(P0)
  , m_state(X0)
  , m_weights(P0.getRows(), Q.getRows(), alphas)
{
  m_cholQ = Q.cholesky().transpose();
  m_d = P0.getRows();
  m_q = Q.getRows();
  m_l = R.getRows();
  m_Id_d.eye(m_d);
}

void vpUnscentedKalmanPose::filter(const vpColVector &omega, const vpColVector &y, const double &dt)
{
  predict(omega, dt);
  update(y, dt);
}

void vpUnscentedKalmanPose::predict(const vpColVector &omega, const double &dt)
{
  vpMatrix P = m_P + TOL * m_Id_d;

  // Update mean
  vpColVector w(m_q, 0.);
  State newState = m_f(m_state, omega, w, dt);

  // // Compute covariance w.r.t state uncertainty
  Weights::Weight w_d = m_weights.m_d;

  // Set sigma points
  vpMatrix xis = w_d.m_sqrtLambda * P.cholesky().transpose();
  std::vector<vpColVector> newXis(2*m_d, vpColVector(m_d, 0.));

  // Retract  sigma points onto the manifold
  vpColVector mean(m_d, 0.);
  for (unsigned int j = 0; j < m_d; ++j) {
    State s_j_p = m_phi(m_state, xis.getRow(j).transpose(), dt);
    State s_j_m = m_phi(m_state, -1. * xis.getRow(j).transpose(), dt);
    State new_s_j_p = m_f(s_j_p, omega, w, dt);
    State new_s_j_m = m_f(s_j_m, omega, w, dt);
    newXis[j] = m_phiinv(newState, new_s_j_p, dt);
    newXis[j + m_d] = m_phiinv(newState, new_s_j_m, dt);
    mean += newXis[j] * w_d.m_wj;
    mean += newXis[j + m_d] * w_d.m_wj;
  }

  // Compute covariance
  vpMatrix newP = w_d.m_w0 * (mean * mean.transpose());
  for (unsigned int j = 0; j < m_d; ++j) {
    newXis[j] = newXis[j] - mean;
    newXis[j + m_d] = newXis[j + m_d] - mean;
    newP += w_d.m_wj * (newXis[j] * newXis[j].transpose());
    newP += w_d.m_wj * (newXis[j + m_d] * newXis[j + m_d].transpose());
  }

  // // Compute covariance w.r.t. noise
  Weights::Weight w_q = m_weights.m_q;
  std::vector<vpColVector> newXisNoise(2 * m_q, vpColVector(m_q, 0.));

  // Retract sigma points onto the manifold
  vpColVector meanNoise(m_q, 0.);
  for (unsigned int j = 0; j < m_q; ++j) {
    vpColVector w_p = w_q.m_sqrtLambda * m_cholQ.getRow(j).transpose();
    vpColVector w_m = -1. * w_q.m_sqrtLambda * m_cholQ.getRow(j).transpose();
    State new_s_j_p = m_f(m_state, omega, w_p, dt);
    State new_s_j_m = m_f(m_state, omega, w_m, dt);
    newXisNoise[j] = m_phiinv(newState, new_s_j_p, dt);
    newXisNoise[j + m_q] = m_phiinv(newState, new_s_j_m, dt);
    meanNoise += w_q.m_wj * newXisNoise[j];
    meanNoise += w_q.m_wj * newXisNoise[j + m_q];
  }

  // Compute covariance
  vpMatrix Q = w_q.m_w0 * (meanNoise * meanNoise.transpose());
  for (unsigned int j = 0; j < m_q; ++j) {
    newXisNoise[j] = newXisNoise[j] - meanNoise;
    newXisNoise[j + m_q] = newXisNoise[j + m_q] - meanNoise;
    Q = w_q.m_wj * (newXisNoise[j] * newXisNoise[j].transpose());
    Q = w_q.m_wj * (newXisNoise[j + m_q] * newXisNoise[j + m_q].transpose());
  }

  // // Update covariance and state
  m_P = newP + Q;
  m_state = newState;
}

void vpUnscentedKalmanPose::update(const vpColVector &y, const double &dt)
{

  vpMatrix P = m_P + TOL * m_Id_d;

  // Set sigma points
  Weights::Weight w_d = m_weights.m_d;
  vpMatrix Pchol = P.cholesky();
  vpMatrix xis = w_d.m_sqrtLambda * Pchol.transpose();

  // // Compute measurement sigma points
  vpMatrix ys(2 * m_d, m_l, 0.);
  vpColVector hat_y = m_h(m_state); // State projected in the observation space
  vpColVector y_bar = w_d.m_wm * hat_y; // Measurement mean
  for (unsigned int j = 0; j < m_d; ++j) {
    State s_j_p = m_phi(m_state, xis.getRow(j).transpose(), dt);
    State s_j_m = m_phi(m_state, -1. * xis.getRow(j).transpose(), dt);
    vpColVector h_sjp = m_h(s_j_p);
    vpColVector h_sjm = m_h(s_j_m);
    ys.insert(h_sjp.transpose(), j, 0);
    ys.insert(h_sjm.transpose(), j + m_d, 0);
    y_bar += w_d.m_wj * ys.getRow(j).transpose();
    y_bar += w_d.m_wj * ys.getRow(j + m_d).transpose();
  }

  // Prune mean before computing covariance
  hat_y = hat_y - y_bar;

  // Covariance computation
  vpMatrix temp(xis.getRows(), 2 * xis.getCols());
  temp.insert(xis, 0, 0);
  temp.insert(-1. * xis, 0, xis.getCols());
  vpMatrix Pxiy = w_d.m_wj * temp * ys;
  vpMatrix Pyy = w_d.m_w0 * (hat_y * hat_y.transpose()) + m_R;
  for (unsigned int j = 0; j < m_d; ++j) {
    Pyy += w_d.m_wj * (ys.getRow(j).transpose() * ys.getRow(j));
    Pyy += w_d.m_wj * (ys.getRow(j + m_d).transpose() * ys.getRow(j + m_d));
  }

  // Kalman gain
  vpMatrix K = Pxiy * Pyy.inverseByCholesky();

  // Update state
  vpColVector xiPlus = K * (y - y_bar);
  m_state = m_phi(m_state, xiPlus, dt);

  // Update covariance
  m_P = P - K * Pyy * K.transpose();

  // Avoid non-symmetric matrix
  m_P = (m_P + m_P.transpose()) / 2.;
}

vpColVector vpUnscentedKalmanPose::asPositionVector(const State &H)
{
  vpTranslationVector t = H.getTranslationVector();
  return asColVector(t);
}

vpColVector vpUnscentedKalmanPose::asColVector(const vpTranslationVector &t)
{
  vpColVector tAsVec(3);
  tAsVec[0] = t[0];
  tAsVec[1] = t[1];
  tAsVec[2] = t[2];
  return tAsVec;
}

vpUnscentedKalmanPose::State vpUnscentedKalmanPose::phiSE3(const vpUnscentedKalmanPose::State &chi, const vpColVector &epsilon, const double &dt)
{
  vpUnscentedKalmanPose::State expEpsilon = vpExponentialMap::direct(epsilon, dt);
  return chi * expEpsilon;
}

vpColVector vpUnscentedKalmanPose::phiinvSE3(const vpUnscentedKalmanPose::State &state, const vpUnscentedKalmanPose::State &hat_state, const double &dt)
{
  return vpExponentialMap::inverse(state.inverse() * hat_state, dt);
}

vpUnscentedKalmanPose::State vpUnscentedKalmanPose::fSE3(const vpUnscentedKalmanPose::State &state, const vpColVector &omega, const vpColVector &w, const double &dt)
{
  return state * vpExponentialMap::direct(omega + w, dt);
}

vpColVector vpUnscentedKalmanPose::hSE3(const vpUnscentedKalmanPose::State &state)
{
  vpColVector pAsVec = asPositionVector(state);
  return pAsVec;
}
END_VISP_NAMESPACE
#else
void vpUnscentedKalman_dummy()
{

}
#endif
