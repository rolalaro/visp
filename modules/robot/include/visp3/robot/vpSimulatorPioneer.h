/*
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2023 by Inria. All rights reserved.
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
 * Pioneer mobile robot simulator without display.
 */

#ifndef vpSimulatorPioneer_H
#define vpSimulatorPioneer_H

/*!
 * \file vpSimulatorPioneer.h
 * \brief class that defines the Pioneer mobile robot simulator equipped with a
 * static camera.
 */

#include <visp3/core/vpColVector.h>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMatrix.h>
#include <visp3/robot/vpPioneer.h>
#include <visp3/robot/vpRobot.h>
#include <visp3/robot/vpRobotSimulator.h>

BEGIN_VISP_NAMESPACE
/*!
 * \class vpSimulatorPioneer
 *
 * \ingroup group_robot_simu_unicycle
 *
 * \brief Class that defines the Pioneer mobile robot simulator equipped with a
 * static camera.
 *
 * It intends to simulate the mobile robot described in vpPioneer class.
 * This robot has 2 dof: \f$(v_x, w_z)\f$, the translational and
 * rotational velocities that are applied at point E.
 *
 * The robot position evolves with respect to a world frame; wMc. When a new
 * joint velocity is applied to the robot using setVelocity(), the position of
 * the camera wrt the world frame is updated.
 *
 * \image html pioneer.png
 *
 * The following code shows how to control this robot in position and velocity.
 * \code
 * #include <visp3/robot/vpSimulatorPioneer.h>
 *
 * #ifdef ENABLE_VISP_NAMESPACE
 * using namespace VISP_NAMESPACE_NAME;
 * #endif
 *
 * int main()
 * {
 *   vpHomogeneousMatrix wMc;
 *   vpSimulatorPioneer robot;
 *
 *   robot.getPosition(wMc); // Position of the camera in the world frame
 *   std::cout << "Default position of the camera in the world frame wMc:\n" << wMc << std::endl;
 *
 *   robot.setSamplingTime(0.100); // Modify the default sampling time to 0.1 second
 *   robot.setMaxTranslationVelocity(1.); // vx max set to 1 m/s
 *   robot.setMaxRotationVelocity(vpMath::rad(90)); // wz max set to 90 deg/s
 *
 *   vpColVector v(2); // we control vx and wz dof
 *   v = 0;
 *   v[0] = 1.; // set vx to 1 m/s
 *   robot.setVelocity(vpRobot::ARTICULAR_FRAME, v);
 *   // The robot has moved from 0.1 meters along the z axis
 *   robot.getPosition(wMc); // Position of the camera in the world frame
 *   std::cout << "New position of the camera wMc:\n" << wMc << std::endl;
 * }
 * \endcode
 *
 * The usage of this class is also highlighted in \ref
 * tutorial-simu-robot-pioneer.
*/
class VISP_EXPORT vpSimulatorPioneer : public vpPioneer, public vpRobotSimulator
{

protected:
  // world to camera
  vpHomogeneousMatrix wMc_;
  // world to end effector frame which is also the mobile
  // robot frame located between the two wheels
  vpHomogeneousMatrix wMe_;
  // cMe_ is a protected member of vpUnicycle

  double xm_;
  double ym_;
  double theta_;

public:
  vpSimulatorPioneer();

public:
  /** @name Inherited functionalities from vpSimulatorPioneer */
  //@{
  void get_eJe(vpMatrix &eJe) VP_OVERRIDE;

  void getPosition(vpHomogeneousMatrix &wMc) const;
  void getPosition(const vpRobot::vpControlFrameType frame, vpColVector &q) VP_OVERRIDE;
  void setVelocity(const vpRobot::vpControlFrameType frame, const vpColVector &vel) VP_OVERRIDE;
  //@}

private:
  void init() VP_OVERRIDE;

  // Non implemented virtual pure functions
  void get_fJe(vpMatrix & /*_fJe */) VP_OVERRIDE { }
  void getDisplacement(const vpRobot::vpControlFrameType /* frame */, vpColVector & /* q */) VP_OVERRIDE { }
  void setPosition(const vpRobot::vpControlFrameType /* frame */, const vpColVector & /* q */) VP_OVERRIDE { }
};
END_VISP_NAMESPACE
#endif
