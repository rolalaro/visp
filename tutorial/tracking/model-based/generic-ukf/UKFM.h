#ifndef UKFM_H
#define UKFM_H

#include <visp3/core/vpConfig.h>
#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
#include <functional>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMatrix.h>

BEGIN_VISP_NAMESPACE
/**
 * \brief This class implements an UKF to filter pose on SE(3) based on velocity and 3D position of an object.
 * To see the original publication and Python code, please refer to \ref{brossard2019Code}
 */
  class UKFM
{
public:
  typedef vpHomogeneousMatrix State; /*!< Internal state of the UKF*/
  typedef std::function<State(const State &, const vpColVector &, const vpColVector &, const double &)> ProcessFunction; /*!< Process fi,ction that projects the state in the future*/
  typedef std::function<vpColVector(const State &)> ObservationFunction; /*!< Observation function that projects the state in the measurement space.*/
  typedef std::function<State(const State &, const vpColVector &, const double &)> RetractationFunction; /*!< Retraction function that apply an object that belong to the Lie's algebra \f[ \boldsymbol{v} \in se(3) \f]  to an object \f[ \boldsymbol{\chi} \in SE(3) \f] by left multiplication \f[ \boldsymbol{\chi}_{n + 1} = \boldsymbol{\chi}_{n} exp(\boldsymbol{v}) \f] */
  typedef std::function<vpColVector(const State &, const State &, const double &)> InverseRetractationFunction; /*!< Inverse retraction function that belong to the Lie's algebra \f[ \boldsymbol{v} \in se(3) \f]  from objects \f[ \boldsymbol{\chi} , \overline{\boldsymbol{\chi}} \in SE(3) \f] using the logarithm function \f[ \boldsymbol{v} = log(\boldsymbol{\chi}^{-1} \overline{\boldsymbol{\chi}}) \f]*/

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
  UKFM(const ProcessFunction &f, const ObservationFunction &h, const RetractationFunction &phi, const InverseRetractationFunction &phi_inv,
      const vpMatrix &Q, const vpMatrix &R, const std::vector<double> &alphas,
      const State &X0, const vpMatrix &P0);

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
