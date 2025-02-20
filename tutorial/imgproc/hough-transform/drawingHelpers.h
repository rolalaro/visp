#ifndef DRAWING_HELPERS_H
#define DRAWING_HELPERS_H

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace drawingHelpers
{
#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

bool display(vpImage<vpRGBa> &I, const std::string &title, const bool &blockingMode);
bool display(vpImage<unsigned char> &I, vpImage<vpRGBa> &Idisp, const std::string &title, const bool &blockingMode);
bool display(vpImage<double> &D, vpImage<vpRGBa> &Idisp, const std::string &title, const bool &blockingMode);
}

#endif
#endif
