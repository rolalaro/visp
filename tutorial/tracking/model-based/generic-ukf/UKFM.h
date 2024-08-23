#ifndef UKFM_H
#define UKFM_H

#include <functional>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMatrix.h>

BEGIN_VISP_NAMESPACE
class UKFM
{
public:
  typedef vpHomogeneousMatrix State;
  typedef std::function<State(const State &, const vpColVector &, const vpColVector &, const double &)> ProcessFunction;
  typedef std::function<vpColVector(const State &)> ObservationFunction;
  typedef std::function<State(const State &, const vpColVector &, const double &)> RetractationFunction;
  typedef std::function<vpColVector(const State &, const State &, const double &)> InverseRetractationFunction;

  UKFM(const ProcessFunction &f, const ObservationFunction &h, const RetractationFunction &phi, const InverseRetractationFunction &phi_inv,
      const vpMatrix &Q, const vpMatrix &R, const std::vector<double> &alphas,
      const State &X0, const vpMatrix &P0);

  void propagation(const vpColVector &omega, const double &dt);

  void update(const vpColVector &y, const double &dt);

  inline void setX0(const State &X0)
  {
    m_state = X0;
  }

  inline State getXest() const
  {
    return m_state;
  }
private:
  struct Weights
  {
    struct Weight
    {
    public:
      Weight(const unsigned int &size, const double &alpha)
      {
        m_lambda = (alpha * alpha - 1.) * size;
        m_sqrtLambda = std::sqrt(m_lambda + size);
        m_wj = 1. / (2. * (size + m_lambda));
        m_wm = m_lambda / (m_lambda + size);
        m_w0 = m_lambda / (m_lambda + size) + 3 - alpha * alpha;
      }

      double m_lambda;
      double m_sqrtLambda;
      double m_wj; /*!< Weights that are common for the computation of the mean and covariance matrices.*/
      double m_wm; /*!< Weight associated to the state for the computation of the mean.*/
      double m_w0; /*!< Weight associated to the first sigma point for the computation of the covariance matrices.*/
    };

    Weight m_d; /*!< Propagation w.r.t. state*/
    Weight m_q; /*!< Propagation w.r.t. noise*/
    Weight m_u; /*!< Update w.r.t. state*/

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
