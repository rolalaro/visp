/*
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2025 by Inria. All rights reserved.
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
 * 2D normalized gravity center moment descriptor (usually described by the
 * pair Xn,Yn)
 */

/*!
  \file vpMomentGravityCenterNormalized.h
  \brief 2D normalized gravity center moment descriptor (usually described by
  the pair Xn,Yn)
*/
#ifndef VP_MOMENT_GRAVITY_CENTER_NORMALIZED_H
#define VP_MOMENT_GRAVITY_CENTER_NORMALIZED_H

#include <visp3/core/vpMomentDatabase.h>
#include <visp3/core/vpMomentGravityCenter.h>

BEGIN_VISP_NAMESPACE
class vpMomentObject;

/*!
 * \class vpMomentGravityCenterNormalized
 *
 * \ingroup group_core_moments
 *
 * \brief Class describing 2D normalized gravity center moment.
 *
 * Centered and normalized gravity center moment is defined as follows:
 * \f$(x_n,y_n)\f$ where \f$x_n = x_g a_n\f$ and \f$y_n = y_g a_n\f$.
 *
 * vpMomentGravityCenterNormalized depends on vpMomentAreaNormalized to get
 * access to \f$a_n\f$ and on vpMomentGravityCenter to get access to
 * \f$(x_g,y_g)\f$ .
*/
class VISP_EXPORT vpMomentGravityCenterNormalized : public vpMomentGravityCenter
{
public:
  vpMomentGravityCenterNormalized();
  void compute() VP_OVERRIDE;
  //! Moment name.
  const std::string name() const VP_OVERRIDE { return "vpMomentGravityCenterNormalized"; }
  void printDependencies(std::ostream &os) const VP_OVERRIDE;
  friend VISP_EXPORT std::ostream &operator<<(std::ostream &os, const vpMomentGravityCenterNormalized &v);
};
END_VISP_NAMESPACE

#endif
