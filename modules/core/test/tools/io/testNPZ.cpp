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
 * Description:
 * Test visp::cnpy::npz_load() / visp::cnpy::npy_save() functions.
 *
*****************************************************************************/

#include <iostream>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpEndian.h>

#if defined(VISP_HAVE_CATCH2) && \
  (defined(_WIN32) || (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))) && \
  defined(VISP_LITTLE_ENDIAN)
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpImage.h>

namespace
{
std::string createTmpDir()
{
  std::string username;
  vpIoTools::getUserName(username);

#if defined(_WIN32)
  std::string tmp_dir = "C:/temp/" + username;
#elif (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
  std::string tmp_dir = "/tmp/" + username;
#endif
  std::string directory_filename = tmp_dir + "/testNPZ/";

  vpIoTools::makeDirectory(directory_filename);
  return directory_filename;
}
}

TEST_CASE("Test visp::cnpy::npy_load/npz_save", "[visp::cnpy I/O]")
{
  std::string directory_filename = createTmpDir();
  REQUIRE(vpIoTools::checkDirectory(directory_filename));
  std::string npz_filename = directory_filename + "/test_npz_read_write.npz";

  SECTION("Read/Save string data")
  {
    const std::string save_string = "Open Source Visual Servoing Platform";
    std::vector<char> vec_save_string(save_string.begin(), save_string.end());
    const std::string identifier = "String";
    visp::cnpy::npz_save(npz_filename, identifier, &vec_save_string[0], { vec_save_string.size() }, "w");

    visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
    REQUIRE(npz_data.find(identifier) != npz_data.end());

    visp::cnpy::NpyArray arr_string_data = npz_data[identifier];
    std::vector<char> vec_arr_string_data = arr_string_data.as_vec<char>();
    // For null-terminated character handling, see:
    // https://stackoverflow.com/a/8247804
    // https://stackoverflow.com/a/45491652
    const std::string read_string = std::string(vec_arr_string_data.begin(), vec_arr_string_data.end());
    CHECK(save_string == read_string);
  }

  SECTION("Read/Save multi-dimensional array")
  {
    size_t height = 5, width = 7, channels = 3;
    std::vector<int> save_vec;
    save_vec.reserve(height*width*channels);
    for (size_t i = 0; i < height*width*channels; i++) {
      save_vec.push_back(i);
    }

    const std::string identifier = "Array";
    visp::cnpy::npz_save(npz_filename, identifier, &save_vec[0], { height, width, channels }, "a"); // append
    visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
    REQUIRE(npz_data.find(identifier) != npz_data.end());

    visp::cnpy::NpyArray arr_vec_data = npz_data[identifier];
    std::vector<int> read_vec = arr_vec_data.as_vec<int>();

    REQUIRE(save_vec.size() == read_vec.size());
    for (size_t i = 0; i < read_vec.size(); i++) {
      CHECK(save_vec[i] == read_vec[i]);
    }
  }

  SECTION("Read/Save vpImage<vpRGBa>")
  {
    vpImage<vpRGBa> I_save(11, 17);
    for (unsigned int i = 0; i < I_save.getRows(); i++) {
      for (unsigned int j = 0; j < I_save.getCols(); j++) {
        I_save[i][j].R = 4 * (i*I_save.getCols() + j) + 0;
        I_save[i][j].G = 4 * (i*I_save.getCols() + j) + 1;
        I_save[i][j].B = 4 * (i*I_save.getCols() + j) + 2;
        I_save[i][j].A = 4 * (i*I_save.getCols() + j) + 3;
      }
    }

    const std::string identifier = "vpImage<vpRGBa>";
    visp::cnpy::npz_save(npz_filename, identifier, &I_save.bitmap[0], { I_save.getRows(), I_save.getCols() }, "a"); // append

    visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
    REQUIRE(npz_data.find(identifier) != npz_data.end());

    visp::cnpy::NpyArray arr_vec_data = npz_data[identifier];
    const bool copy_data = false;
    vpImage<vpRGBa> I_read(arr_vec_data.data<vpRGBa>(), arr_vec_data.shape[0], arr_vec_data.shape[1], copy_data);

    CHECK(I_save.getSize() == I_read.getSize());
    CHECK(I_save == I_read);
  }

  SECTION("Read/Save std::vector<vpColVector>")
  {
    std::vector<vpColVector> vec_colVector_save;
    for (size_t i = 0; i < 7; i++) {
      vpColVector vector(3);
      vector[0] = i*3 + 0;
      vector[1] = i*3 + 1;
      vector[2] = i*3 + 2;
      vec_colVector_save.push_back(vector);
    }

    const std::string identifier = "std::vector<vpColVector>";
    visp::cnpy::npz_save(npz_filename, identifier, &vec_colVector_save[0], { vec_colVector_save.size() }, "a"); // append

    visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
    REQUIRE(npz_data.find(identifier) != npz_data.end());

    visp::cnpy::NpyArray arr_vec_data = npz_data[identifier];
    std::vector<vpColVector> vec_colVector_read = arr_vec_data.as_vec<vpColVector>();

    REQUIRE(vec_colVector_save.size() == vec_colVector_read.size());
    for (size_t i = 0; i < vec_colVector_save.size(); i++) {
      CHECK(vec_colVector_save[i] == vec_colVector_read[i]);
    }
  }

  SECTION("Read/Save std::vector<vpHomogeneousMatrix>")
  {
    std::vector<vpHomogeneousMatrix> vec_homogeneousMatrix_save;
    for (size_t i = 0; i < 7; i++) {
      vpColVector u(3);
      u[0] = i*4 + 0;
      u[1] = i*4 + 1;
      u[2] = i*4 + 2;
      u.normalize();
      double theta = vpMath::rad(5*i + 2);
      vpThetaUVector tu(std::vector<double>{ theta *u[0], theta *u[1], theta *u[2] });
      // std::cout << "theta=" << tu.getTheta() << " ; u=" << tu.getU().transpose() << std::endl;

      vpTranslationVector trans(i*4 + 10, i*4 + 20, i*4 + 30);
      vpHomogeneousMatrix cMo(trans, tu);
      // std::cout << "cMo:\n" << cMo << std::endl;
      vec_homogeneousMatrix_save.push_back(cMo);
    }

    const std::string identifier = "std::vector<vpHomogeneousMatrix>";
    visp::cnpy::npz_save(npz_filename, identifier, &vec_homogeneousMatrix_save[0],
      { vec_homogeneousMatrix_save.size() }, "a"); // append

    visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
    REQUIRE(npz_data.find(identifier) != npz_data.end());

    visp::cnpy::NpyArray arr_vec_data = npz_data[identifier];
    std::vector<vpHomogeneousMatrix> vec_homogeneousMatrix_read = arr_vec_data.as_vec<vpHomogeneousMatrix>();

    REQUIRE(vec_homogeneousMatrix_save.size() == vec_homogeneousMatrix_read.size());
    for (size_t i = 0; i < vec_homogeneousMatrix_save.size(); i++) {
      // std::cout << "cMo save:\n" << vec_homogeneousMatrix_save[i]
      //   << "\ncMo read:\n" << vec_homogeneousMatrix_read[i] << std::endl;
      CHECK(vec_homogeneousMatrix_save[i] == vec_homogeneousMatrix_read[i]);
    }
  }

  REQUIRE(vpIoTools::remove(directory_filename));
  REQUIRE(!vpIoTools::checkDirectory(directory_filename));
}

// https://en.cppreference.com/w/cpp/types/integer
// https://github.com/catchorg/Catch2/blob/devel/docs/test-cases-and-sections.md#type-parametrised-test-cases
using BasicTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double>;
TEMPLATE_LIST_TEST_CASE("Test visp::cnpy::npy_load/npz_save", "[BasicTypes][list]", BasicTypes)
{
  std::string directory_filename = createTmpDir();
  REQUIRE(vpIoTools::checkDirectory(directory_filename));
  std::string npz_filename = directory_filename + "/test_npz_read_write.npz";

  const std::string identifier = "data";
  TestType save_data = std::numeric_limits<TestType>::min();
  visp::cnpy::npz_save(npz_filename, identifier, &save_data, { 1 }, "w");

  visp::cnpy::npz_t npz_data = visp::cnpy::npz_load(npz_filename);
  REQUIRE(npz_data.find(identifier) != npz_data.end());
  visp::cnpy::NpyArray arr_data = npz_data[identifier];
  TestType read_data = *arr_data.data<TestType>();
  CHECK(save_data == read_data);

  save_data = std::numeric_limits<TestType>::max();
  visp::cnpy::npz_save(npz_filename, identifier, &save_data, { 1 }, "a"); // append

  npz_data = visp::cnpy::npz_load(npz_filename);
  REQUIRE(npz_data.find(identifier) != npz_data.end());
  arr_data = npz_data[identifier];
  read_data = *arr_data.data<TestType>();
  CHECK(save_data == read_data);

  REQUIRE(vpIoTools::remove(directory_filename));
  REQUIRE(!vpIoTools::checkDirectory(directory_filename));
}

int main(int argc, char *argv[])
{
  Catch::Session session; // There must be exactly one instance

  // Let Catch (using Clara) parse the command line
  session.applyCommandLine(argc, argv);

  int numFailed = session.run();

  // numFailed is clamped to 255 as some unices only use the lower 8 bits.
  // This clamping has already been applied, so just return it here
  // You can also do any post run clean-up here
  return numFailed;
}
#else
int main() { return EXIT_SUCCESS; }
#endif
