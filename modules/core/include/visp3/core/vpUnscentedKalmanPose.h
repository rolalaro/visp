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
 * Unscented Kalman Filter to filter pose on SE(3)
 */

#ifndef VP_UNSCENTED_KALMAN_POSE_H
#define VP_UNSCENTED_KALMAN_POSE_H

#include <visp3/core/vpConfig.h>

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpExponentialMap.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMatrix.h>

BEGIN_VISP_NAMESPACE
/*!
  \class vpUnscentedKalmanPose
  \ingroup group_core_kalman
  This class permits to use Unscented Kalman Filter (UKF) to tackle pose on SE(3) filtering
*/
class VISP_EXPORT vpUnscentedKalmanPose
{
public:
  /**
   * \brief Construct a new vpUnscentedKalmanPose object.
   *
   * \param[in] Q The covariance introduced by performing the prediction step.
   * \param[in] R The covariance introduced by performing the update step.
   * \param[in] muNoiseMeas The mean of the measurement noise.
   * \param[in] alphaPred The parameter for the drawing of the sigma points during the predict step.
   * \param[in] alphaUpdate The parameter for the drawing of the sigma points during the update step.
   */
  vpUnscentedKalmanPose(const vpMatrix &Q, const vpMatrix &R, const vpColVector &muNoiseMeas, const double &alphaPred, const double &alphaUpdate);

  /**
   * \brief Set the guess of the initial state and velocity.
   *
   * \param[in] X0 Guess of the initial state.
   * \param[in] P0 Guess of the initial state covariance matrix.
   * \param[in] omega0 Guess of the initial velocity.
   */
  void init(const vpHomogeneousMatrix &X0, const vpMatrix &P0, const vpColVector &omega0);

  /**
   * \brief Permit to change the covariance introduced at each prediction step.
   *
   * \param[in] Q The process covariance matrix.
   */
  inline void setProcessCovariance(const vpMatrix &Q)
  {
    m_Q = Q;
  }

  /**
   * \brief Permit to change the covariance introduced at each update step.
   *
   * \param[in] R The measurement covariance matrix.
   */
  inline void setMeasurementCovariance(const vpMatrix &R)
  {
    m_R = R;
  }

  /**
   * \brief Perform first the prediction step and then the filtering step.
   *
   * \param[in] z The new measurement.
   * \param[in] dt The time in the future we must predict.
   */
  void filter(const vpColVector &z, const double &dt);

  /**
   * \brief Get the estimated (i.e. filtered) covariance of the state.
   *
   * \return vpMatrix The filtered covariance matrix.
   */
  inline vpMatrix getPest() const
  {
    return m_Pest;
  }

  /**
   * \brief Get the predicted covariance of the state, i.e. the covariance of the prior.
   *
   * \return vpMatrix The predicted covariance matrix.
   */
  inline vpMatrix getPpred() const
  {
    return m_Ppred;
  }

  /**
   * \brief Get the estimated (i.e. filtered) state.
   *
   * \return vpHomogeneousMatrix The estimated state.
   */
  inline vpHomogeneousMatrix getXest() const
  {
    return m_Xest;
  }

  /**
   * \brief Get the predicted state (i.e. the prior).
   *
   * \return vpColVector The predicted state.
   */
  inline vpHomogeneousMatrix getXpred() const
  {
    return m_Xpred;
  }

  inline vpColVector getOmega() const
  {
    return m_omega;
  }

private:
  const unsigned int m_q = 6; /*!< Dimension of the state. It is a pose, so it is 6.*/
  const unsigned int m_k = 6; /*!< Dimension of the measurements. They are poses, so it is 6.*/

  /// Members related to the predict step
  double m_alphaPredict; /*!< Scale parameter for the sigma points drawing during the prediction step.*/
  vpColVector m_omega; /*!< The velocity at the current step.*/
  vpMatrix m_Q; /*!< The covariance introduced by performing the prediction step.*/
  vpHomogeneousMatrix m_Xpred; /*!< The predicted state, i.e. the mean of the prior.*/
  vpMatrix m_Ppred; /*!< The covariance matrix of the prior.*/

  /// Members related to the update step
  double m_alphaUpdate; /*!< Scale parameter for the sigma points drawing during the update step.*/
  vpColVector m_muNoiseMeas; /*!< The mean of the measurements noise.*/
  vpMatrix m_R; /*!< The covariance introduced by performing the update step.*/
  vpMatrix m_Pz; /*!< The covariance matrix of the measurement sigma points.*/
  vpMatrix m_Pxz; /*!< The cross variance of the state and the measurements.*/
  vpMatrix m_K; /*!< The Kalman gain.*/
  vpHomogeneousMatrix m_Xest; /*!< The estimated (i.e. filtered) state variables.*/
  vpMatrix m_Pest; /*!< The estimated (i.e. filtered) covariance matrix.*/

  /**
   * \brief Predict the new state based on the last state and how far in time we want to predict.
   *
   * \param[in] dt The time in the future we must predict.
   */
  void predict(const double &dt);

  /**
   * \brief Update the estimate of the state based on a new measurement.
   *
   * \param[in] z The measurements at the current timestep.
   * \param[in] dt The period since the last update.
   */
  void update(const vpColVector &z, const double &dt);

  /**
   * \brief Structure that stores the results of the unscented transform.
   */
  typedef struct vpSigmaPointDrawingResult
  {
    std::vector<vpColVector> m_chis; /*!< The sigma points.*/
    vpColVector m_muchis; /*!< The mean of the sigma points.*/
    std::vector<double> m_wm; /*!< The points for the mean computation.*/
    std::vector<double> m_wc; /*!< The points for the covariance computation.*/
  } vpSigmaPointDrawingResult;

  /**
   * \brief Draw the sigma points for the predict step.
   */
  vpSigmaPointDrawingResult sigmaPointsDrawingPredict();

  /**
   * \brief Compute the unscented transform of the sigma points, leading
   * to the new predictions of the state and of the state covariance.
   *
   * \param[in] muprev The previous mean.
   * \param[in] sigmaPoints The sigma points we consider.
   * \param[in] wc The weights to apply for the covariance computation.
   * \param[in] omega The velocity at the current step.
   * \param[in] dt The ellapsed time since the last call.
   */
  void unscentedTransformPredict(const vpHomogeneousMatrix &muprev, const std::vector<vpColVector> &sigmaPoints,
    const std::vector<double> &wc, const vpColVector &omega, const double &dt);

  /**
   * \brief Draw the sigma points for the update step.
   *
   * \return vpSigmaPointDrawingResult The sigma points along with other useful info.
   */
  vpSigmaPointDrawingResult sigmaPointsDrawingUpdate();

  /**
   * \brief Structure that stores the results of the unscented transform.
   */
  typedef struct vpUnscentedTransformUpdateResult
  {
    vpColVector m_mu; /*!< The mean of the chi points.*/
    std::vector<vpColVector> m_z; /*!< The chi points projected in the measurement space.*/
    vpMatrix m_P; /*!< The covariance matrix \f$P_{zz}\f$.*/
  } vpUnscentedTransformResult;

  /**
   * \brief Compute the unscented transform of the sigma points.
   *
   * \param[in] sigmaPoints The sigma points we consider.
   * \param[in] wm The weights to apply for the mean computation.
   * \param[in] wc The weights to apply for the covariance computation.
   * \param[in] dt The period since the last update.
   * \return vpUnscentedTransformResult The mean and covariance of the sigma points.
   */
  vpUnscentedTransformResult unscentedTransformUpdate(const std::vector<vpColVector> &sigmaPoints,
    const std::vector<double> &wm, const std::vector<double> &wc, const double &dt);
};
END_VISP_NAMESPACE
#endif
#endif
