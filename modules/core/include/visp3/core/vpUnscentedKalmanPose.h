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
  This class permits to use Unscented Kalman Filter (UKF) to tackle pose on SE(3) filtering.
  To see the original publication and Python code, please refer to \ref{brossard2019Code}.
*/
class VISP_EXPORT vpUnscentedKalmanPose
{
public:
  typedef vpHomogeneousMatrix State; /*!< Internal state of the UKF*/
  typedef std::function<State(const State &, const vpColVector &, const vpColVector &, const double &)> ProcessFunction; /*!< Process fi,ction that projects the state in the future*/
  typedef std::function<vpColVector(const State &)> ObservationFunction; /*!< Observation function that projects the state in the measurement space.*/
  typedef std::function<State(const State &, const vpColVector &, const double &)> RetractationFunction; /*!< Retraction function that apply an object that belong to the Lie's algebra \f[ \boldsymbol{v} \in se(3) \f]  to an object \f[ \boldsymbol{\chi} \in SE(3) \f]. */
  typedef std::function<vpColVector(const State &, const State &, const double &)> InverseRetractationFunction; /*!< Inverse retraction function that belong to the Lie's algebra \f[ \boldsymbol{v} \in se(3) \f]  from objects \f[ \boldsymbol{\chi} , \overline{\boldsymbol{\chi}} \in SE(3) \f].*/

  /**
   * \brief Construct a new UKFM object.
   *
   * \param[in] f The process function.
   * \param[in] h The observation function.
   * \param[in] phi The retraction function.
   * \param[in] phi_inv The inverse retraction function.
   * \param[in] Q The process noise covariance matrix.
   * \param[in] R The measurement noise covariance matrix.
   * \param[in] alphas The weights for the computation of the sigma points, such as
   * alphas[0]: used in predict for the Unscented Transform of the state,
   * alphas[1]: used in predict for the Unscented Transform of the noise,
   * alphas[2]: used in update for the Unscented Transform of the state,
   * \param[in] X0 The initial estimate of the state.
   * \param[in] P0 The initial estimated state covariance matrix.
   */
  vpUnscentedKalmanPose(const vpMatrix &Q, const vpMatrix &R, const std::vector<double> &alphas,
      const State &X0, const vpMatrix &P0, const ProcessFunction &f = fSE3, const ObservationFunction &h = hSE3, const RetractationFunction &phi = phiSE3, const InverseRetractationFunction &phi_inv = phiinvSE3);

  /**
   * \brief Performs first a predict step and then an
   * update step.
   *
   * \param[in] omega The velocity of the object such as v[0..2] = linear velocity and v[3..5] = angular velocity.
   * \param[in] y The measurements, that correspond to the 3D position of the object.
   * \param[in] dt The period since the last call to either update or predict.
   */
  void filter(const vpColVector &omega, const vpColVector &y, const double &dt);

  /**
   * \brief Project the internal state in the future.
   *
   * \param[in] omega The velocity of the object such as v[0..2] = linear velocity and v[3..5] = angular velocity.
   * \param[in] dt The period since the last call to either update or predict.
   */
  void predict(const vpColVector &omega, const double &dt);

  /**
   * \brief Update the internal state of the UKF to get a filtered state based
   *  on the measurements.
   *
   * \param[in] y The measurements, that correspond to the 3D position of the object.
   * \param dt The period since the last call to either update or predict.
   */
  void update(const vpColVector &y, const double &dt);

  /**
   * \brief Set the initial guess of the state.
   *
   * \param[in] X0 The initial guess of the state.
   */
  inline void setX0(const State &X0)
  {
    m_state = X0;
  }

  /**
   * \brief Get the filtered state or the predicted state depending
   * on which between predict and update has been the last call.
   *
   * @return State
   */
  inline State getState() const
  {
    return m_state;
  }

  /**
   * \brief Transform a state into a vpCoLVector that represetns the 3D position.
   *
   * \param[in] H The state.
   * \return vpColVector The 3D postion that corresponds to the state.
   */
  static vpColVector asPositionVector(const State &t);

  /**
   * \brief Transform a vpTranslationVector into a vpColVector.
   *
   * \param[in] t The translation t that corresponds to a 3D position.
   * \return vpColVector The corresponding 3D position as a vpColVector.
   */
  static vpColVector asColVector(const vpTranslationVector &t);

  /**
   * \brief Default retraction function that apply an object that belong to the Lie's algebra
   * \f[ \boldsymbol{v} \in se(3) \f]  to an object \f[ \boldsymbol{\chi} \in SE(3) \f] by
   * left multiplication \f[ \boldsymbol{\chi}_{n + 1} = \boldsymbol{\chi}_{n} exp(\boldsymbol{v}, dt) \f] .
   *
   * \param chi The state to which the object lying on the Lie's algebra must be applied.
   * \param epsilon The object lying on the Lie's algebra that must be applied.
   * \param dt The period during which epsilon is applied.
   * \return State The updated state.
   */
  static State phiSE3(const State &chi, const vpColVector &epsilon, const double &dt);

  /**
   * \brief Inverse retraction function that belong to the Lie's algebra \f[ \boldsymbol{v} \in se(3) \f]
   * from objects \f[ \boldsymbol{\chi} , \overline{\boldsymbol{\chi}} \in SE(3) \f] using the logarithm
   * function \f[ \boldsymbol{v} = log(\boldsymbol{\chi}^{-1} \overline{\boldsymbol{\chi}}) \f]
   *
   * \param state The state \f[ \boldsymbol{\chi} \f] .
   * \param hat_state The mean of the state \f[ \overline{\boldsymbol{\chi}} \f] .
   * \param dt The period of time that corresponds to this displacement.
   * \return vpColVector The corresponding object in the Lie's algebra.
   */
  static vpColVector phiinvSE3(const State &state, const State &hat_state, const double &dt);

  /**
   * \brief Default process function such as \f[ \chi_{n + 1} = \chi_{n} exp((\boldsymbol{\Omega} + w)dt) \f].
   *
   * \param[in] state \f[ \chi \f]
   * \param[in] omega The velocity, such as \f[ \boldsymbol{\Omega} = (\boldsymbol{v}^T \boldsymbol{\omega}^T)^T \in R^6 \f]
   * \param[in] w The potential noise.
   * \param[in] dt The period during which is applied the velocity vector.
   * \return State The updated state.
   */
  static State fSE3(const State &state, const vpColVector &omega, const vpColVector &w, const double &dt);

  /**
   * \brief The default measurement function such as \f[
     \boldsymbol{T} = \left[ \begin{array}{cc}
     \boldsymbol{R} \boldsymbol{t} \\
     \boldsymbol{0}_ {1 x 3} 1
     \end{array}\right]
     , h(\boldsymbol{T}) = \boldsymbol{t} \in R^3\f]
   *
   * \param state The internal state of the UKF.
   * \return vpColVector The 3D position that corresponds to the state.
   */
  static vpColVector hSE3(const State &state);
private:
  struct Weights
  {
    /**
     * \brief  Structure that implements a single weight for either the
     * computation of the mean or a covariance matrix in the Unscented transform.
     */
    struct Weight
    {
    public:
      /**
       * \brief Construct a new Weight object.
       *
       * \param[in] size Size of the vector to which the Weight will be applied to.
       * \param[in] alpha Coefficient to compute the weight.
       */
      Weight(const unsigned int &size, const double &alpha)
      {
        m_lambda = (alpha * alpha - 1.) * size;
        m_sqrtLambda = std::sqrt(m_lambda + size);
        m_wj = 1. / (2. * (size + m_lambda));
        m_wm = m_lambda / (m_lambda + size);
        m_w0 = m_lambda / (m_lambda + size) + 3 - alpha * alpha;
      }

      double m_lambda; /*!< The modified coefficient used to compute the weight.*/
      double m_sqrtLambda; /*!< Square root of the modified coefficient used to compute the weight.*/
      double m_wj; /*!< Weights that are common for the computation of the mean and covariance matrices.*/
      double m_wm; /*!< Weight associated to the state for the computation of the mean.*/
      double m_w0; /*!< Weight associated to the first sigma point for the computation of the covariance matrices.*/
    };

    Weight m_d; /*!< Predict w.r.t. state weights*/
    Weight m_q; /*!< Predict w.r.t. noise weights*/
    Weight m_u; /*!< Update w.r.t. state weights*/

    /**
     * \brief Construct a new Weights object
     *
     * \param[in] d The size of the state.
     * \param[in] q The size of the noise.
     * \param[in] alphas The coefficients used to compute the weights, such as
     * alphas[0]: used in predict for the Unscented Transform of the state,
     * alphas[1]: used in predict for the Unscented Transform of the noise,
     * alphas[2]: used in update for the Unscented Transform of the state.
     */
    Weights(const unsigned int &d, const unsigned int &q, const std::vector<double> &alphas)
      : m_d(d, alphas[0])
      , m_q(q, alphas[1])
      , m_u(d, alphas[2])
    { }
  };

  static const double TOL; /*!< tolerance parameter (avoid numerical issue)*/
  ProcessFunction m_f; /*!< Process function that projects the state in the future.*/
  ObservationFunction m_h; /*!< Observation function that projects the state in the measurement space.*/
  RetractationFunction m_phi; /*!< Retraction function.*/
  InverseRetractationFunction m_phiinv; /*!< Inverse retraction function.*/
  vpMatrix m_Q; /*!< State noise covariance matrix.*/
  vpMatrix m_R; /*!< Measurement covariance matrix.*/
  vpMatrix m_P; /*!< State covariance matrix.*/
  State m_state; /*!< Estimated state.*/
  vpMatrix m_cholQ; /*!< Cholesky's decomposition of the Q matrix ~= its square root*/
  unsigned int m_d; /*!< Size of the state.*/
  unsigned int m_q; /*!< Size of the noise.*/
  unsigned int m_l; /*!< Size of the measurements.*/
  vpMatrix m_Id_d; /*!< Identity matrix of size d*/
  Weights m_weights; /*!< Weights used for the Unscented Transform.*/
};
END_VISP_NAMESPACE
#endif
#endif
