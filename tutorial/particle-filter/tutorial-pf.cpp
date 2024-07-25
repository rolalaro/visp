/****************************************************************************
 *
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
*****************************************************************************/

//! \example tutorial-pf.cpp

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpTime.h>

#include "vpCommonData.h"
#include "vpTutoSegmentation.h"

int main(const int argc, const char *argv[])
{
#ifdef ENABLE_VISP_NAMESPACE
  using VISP_NAMESPACE_NAME;
#endif
  tutorial::vpCommonData data;
  int returnCode = data.init(argc, argv);
  if (returnCode != tutorial::vpCommonData::SOFTWARE_CONTINUE) {
    return returnCode;
  }

  const double period = 33.; // 33ms period, i.e. 30Hz
  while (!data.m_grabber.end()) {
    double t0 = vpTime::measureTimeMs();
    data.m_grabber.acquire(data.m_I_orig);
    tutorial::performSegmentationHSV(data);
#ifdef VISP_HAVE_DISPLAY
    vpDisplay::display(data.m_I_orig);
    vpDisplay::display(data.m_I_segmented);
#endif
#ifdef VISP_HAVE_DISPLAY
    vpDisplay::flush(data.m_I_orig);
    vpDisplay::flush(data.m_I_segmented);
#endif
    vpTime::wait(t0, period);
  }
  return 0;
}
