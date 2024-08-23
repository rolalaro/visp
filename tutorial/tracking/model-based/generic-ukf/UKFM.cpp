#include "UKFM.h"

void log(const std::string &funcName, const std::string &text)
{
  std::cout << "[" << funcName << "] " << text << std::endl << std::flush;
}

BEGIN_VISP_NAMESPACE
std::string toString(const vpColVector &v)
{
  std::string text("(");
  text += std::to_string(v.getRows()) + std::string(" x 1) : [");
  for (unsigned int i = 0; i < v.getRows(); ++i) {
    text += std::string(" ") + std::to_string(v[i]);
  }
  text += std::string(" ]");
  return text;
}

void logMat(const vpArray2D<double> &M, const unsigned int &nbIndent)
{
  for (unsigned int i = 0; i < M.getRows(); ++i) {
    for (unsigned int k = 0; k < nbIndent; ++k) {
      std::cout << "\t";
    }
    for (unsigned int j = 0; j < M.getCols() - 1; ++j) {
      std::cout << M[i][j] << " ; ";
    }
    std::cout << M[i][M.getCols() - 1] << std::endl;
  }
}

const double  UKFM::TOL = 1e-9;

UKFM::UKFM(const ProcessFunction &f, const ObservationFunction &h, const RetractationFunction &phi, const InverseRetractationFunction &phi_inv,
      const vpMatrix &Q, const vpMatrix &R, const std::vector<double> &alphas,
      const State &X0, const vpMatrix &P0)
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

void UKFM::propagation(const vpColVector &omega, const double &dt)
{
  // log("UKFM::propagation", "Begin ...");
  vpMatrix P = m_P + TOL * m_Id_d;

  // Update mean
  log("UKFM::propagation", "\tUpdating mean ...");
  vpColVector w(m_q, 0.);
  log("UKFM::propagation", "\t\tOmega:");
  logMat(omega.transpose(), 2);
  log("UKFM::propagation", "\t\tPrevious state:");
  logMat(m_state, 2);
  State newState = m_f(m_state, omega, w, dt);
  log("UKFM::propagation", "\t\tUpdated state:");
  logMat(m_state, 2);

  // // Compute covariance w.r.t state uncertainty
  // log("UKFM::propagation", "\tComputing cov w.r.t. state ...");
  Weights::Weight w_d = m_weights.m_d;

  // Set sigma points
  // log("UKFM::propagation", "\t\tSigma points begin...");
  // log("UKFM::propagation", "\t\t\tPchol =");
  // logMat(P.cholesky(), 2);
  // log("UKFM::propagation", "\t\t\tsqrt(lambda) = " + std::to_string(w_d.m_sqrtLambda));
  vpMatrix xis = w_d.m_sqrtLambda * P.cholesky().transpose();
  std::vector<vpColVector> newXis(2*m_d, vpColVector(m_d, 0.));

  // Retract  sigma points onto the manifold
  // log("UKFM::propagation", "\t\t->Retracting chis ...");
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
  // log("UKFM::propagation", "\t\t->Computing cov ...");
  // log("UKFM::propagation", "\t\t\tw0 = " + std::to_string(w_d.m_w0));
  // log("UKFM::propagation", "\t\t\tmean = " +toString(mean));
  vpMatrix newP = w_d.m_w0 * (mean * mean.transpose());
  for (unsigned int j = 0; j < m_d; ++j) {
    newXis[j] = newXis[j] - mean;
    newXis[j + m_d] = newXis[j + m_d] - mean;
    newP += w_d.m_wj * (newXis[j] * newXis[j].transpose());
    newP += w_d.m_wj * (newXis[j + m_d] * newXis[j + m_d].transpose());
  }

  // // Compute covariance w.r.t. noise
  // log("UKFM::propagation", "\tComputing cov w.r.t. noise ...");
  Weights::Weight w_q = m_weights.m_q;
  std::vector<vpColVector> newXisNoise(2 * m_q, vpColVector(m_q, 0.));

  // Retract sigma points onto the manifold
  // log("UKFM::propagation", "\t\t->Retracting chis ...");
  // log("UKFM::propagation", "\t\t\t-> cholQ = ");
  logMat(m_cholQ, 3);
  vpColVector meanNoise(m_q, 0.);
  for (unsigned int j = 0; j < m_q; ++j) {
    vpColVector w_p = w_q.m_sqrtLambda * m_cholQ.getRow(j).transpose();
    vpColVector w_m = -1. * w_q.m_sqrtLambda * m_cholQ.getRow(j).transpose();
    // log("UKFM::propagation", std::string("\t\t\t-> w_p[") + std::to_string(j) + std::string("] = ") + toString(w_p));
    // log("UKFM::propagation", std::string("\t\t\t-> w_m[") + std::to_string(j) + std::string("] = ") + toString(w_m));
    State new_s_j_p = m_f(m_state, omega, w_p, dt);
    State new_s_j_m = m_f(m_state, omega, w_m, dt);
    // log("UKFM::propagation", "\t\t\t-> new_s_j_p = ");
    // logMat(new_s_j_p, 3);
    // log("UKFM::propagation", "\t\t\t-> new_s_j_m = ");
    // logMat(new_s_j_m, 3);
    newXisNoise[j] = m_phiinv(newState, new_s_j_p, dt);
    newXisNoise[j + m_q] = m_phiinv(newState, new_s_j_m, dt);
    // log("UKFM::propagation", std::string("\t\t\t-> newXisNoise[") + std::to_string(j) + std::string("] = "));
    // logMat(newXisNoise[j].transpose(), 3);
    // log("UKFM::propagation", std::string("\t\t\t-> newXisNoise[") + std::to_string(j + m_q) + std::string("] = "));
    // logMat(newXisNoise[j + m_q].transpose(), 3);
    // log("UKFM::propagation", std::string("\t\t\t-> w_q.m_wj = ") + std::to_string(w_q.m_wj));
    meanNoise += w_q.m_wj * newXisNoise[j];
    meanNoise += w_q.m_wj * newXisNoise[j + m_q];
  }

  // Compute covariance
  // log("UKFM::propagation", "\t\t->Computing cov ...");
  // log("UKFM::propagation", std::string("\t\t\t-> meanNoise = ") + toString(meanNoise));
  vpMatrix Q = w_q.m_w0 * (meanNoise * meanNoise.transpose());
  for (unsigned int j = 0; j < m_q; ++j) {
    newXisNoise[j] = newXisNoise[j] - meanNoise;
    newXisNoise[j + m_q] = newXisNoise[j + m_q] - meanNoise;
    Q = w_q.m_wj * (newXisNoise[j] * newXisNoise[j].transpose());
    Q = w_q.m_wj * (newXisNoise[j + m_q] * newXisNoise[j + m_q].transpose());
  }

  // // Update covariance and state
  log("UKFM::propagation", "\tUpdating ...");
  // log("UKFM::propagation", "\t\tnewP = ");
  // logMat(newP, 2);
  // log("UKFM::propagation", "\t\tQ = ");
  // logMat(Q, 2);
  m_P = newP + Q;
  m_state = newState;
  log("UKFM::propagation", "\t\tnewState = ");
  logMat(newState, 2);
  // log("UKFM::propagation", "Done !");
}

void UKFM::update(const vpColVector &y, const double &dt)
{
  // log("UKFM::update", "Begin ...");
  // log("UKFM::update", "\t\tP = ");
  // logMat(m_P, 2);
  // log("UKFM::update", "\t\tI_d = ");
  // logMat(m_Id_d, 2);
  vpMatrix P = m_P + TOL * m_Id_d;

  // Set sigma points
  // log("UKFM::update", "\tSet sigma points ...");
  Weights::Weight w_d = m_weights.m_d;
  // log("UKFM::update", "\t\tsqrt(lambda) = " + std::to_string(w_d.m_sqrtLambda));
  // log("UKFM::update", "\t\tchol(P)^T = ");
  vpMatrix Pchol = P.cholesky();
  // logMat(Pchol.transpose(), 2);
  vpMatrix xis = w_d.m_sqrtLambda * Pchol.transpose();

  // // Compute measurement sigma points
  // log("UKFM::update", "\tCompute measurement sigma points ...");
  vpMatrix ys(2 * m_d, m_l, 0.);
  vpColVector hat_y = m_h(m_state); // State projected in the observation space
  vpColVector y_bar = w_d.m_wm * hat_y; // Measurement mean
  // log("UKFM::update", "\t\tstate = ");
  // logMat(m_state, 2);
  // log("UKFM::update", "\t\txis = ");
  // logMat(xis, 2);
  for (unsigned int j = 0; j < m_d; ++j) {
    // log("UKFM::update", "\t\tComputing retraction of sigma points ...");
    State s_j_p = m_phi(m_state, xis.getRow(j).transpose(), dt);
    State s_j_m = m_phi(m_state, -1. * xis.getRow(j).transpose(), dt);
    // log("UKFM::update", "\t\tInserting  in ys ...");
    vpColVector h_sjp = m_h(s_j_p);
    // log("UKFM::update", std::string("\t\th(sjp)[") + std::to_string(j) + std::string(" ] = ") + toString(h_sjp));
    vpColVector h_sjm = m_h(s_j_m);
    // log("UKFM::update", std::string("\t\th(sjm)[") + std::to_string(j) + std::string(" ] = ") + toString(h_sjm));
    ys.insert(h_sjp.transpose(), j, 0);
    ys.insert(h_sjm.transpose(), j + m_d, 0);
    // log("UKFM::update", "\t\tUpdating ybar ...");
    y_bar += w_d.m_wj * ys.getRow(j).transpose();
    y_bar += w_d.m_wj * ys.getRow(j + m_d).transpose();
  }

  // Prune mean before computing covariance
  // log("UKFM::update", "\tPruning before cov ...");
  hat_y = hat_y - y_bar;

  // Covariance computation
  // log("UKFM::update", "\tComputing cov ...");
  vpMatrix temp(xis.getRows(), 2 * xis.getCols());
  temp.insert(xis, 0, 0);
  temp.insert(-1. * xis, 0, xis.getCols());
  vpMatrix Pxiy = w_d.m_wj * temp * ys;
  vpMatrix Pyy = w_d.m_w0 * (hat_y * hat_y.transpose()) + m_R;
  for (unsigned int j = 0; j < m_d; ++j) {
    // log("UKFM::update", std::string("\t\tys[") + std::to_string(j) + std::string(" ] = ") + toString(ys.getRow(j).transpose()));
    // log("UKFM::update", "\t\twj = " + std::to_string(w_d.m_wj));
    Pyy += w_d.m_wj * (ys.getRow(j).transpose() * ys.getRow(j));
    Pyy += w_d.m_wj * (ys.getRow(j + m_d).transpose() * ys.getRow(j + m_d));
  }

  // Kalman gain
  // log("UKFM::update", "\tComputing K ...");
  // log("UKFM::update", "\t\tPyy = ");
  // logMat(Pyy, 2);
  // log("UKFM::update", "\t\tPxiy = ");
  // logMat(Pxiy, 2);
  vpMatrix K = Pxiy * Pyy.inverseByCholesky();

  // Update state
  log("UKFM::update", "\tUpdating state ...");
  vpColVector xiPlus = K * (y - y_bar);
  m_state = m_phi(m_state, xiPlus, dt);
  log("UKFM::update", "\t\txiPlus = ");
  logMat(xiPlus, 2);

  // Update covariance
  // log("UKFM::update", "\tUpdating cov ...");
  m_P = P - K * Pyy * K.transpose();

  // Avoid non-symmetric matrix
  // log("UKFM::update", "\tAvoiding non-symmetry ...");
  m_P = (m_P + m_P.transpose()) / 2.;

  // log("UKFM::update", "Done !");
}
END_VISP_NAMESPACE
