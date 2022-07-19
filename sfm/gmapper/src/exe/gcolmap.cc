// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "controllers/global_mapper.h"
#include "util/option_manager.h"
#include "util/version.h"

using namespace colmap;

int RunGlobalMapper(int argc, char** argv) {
  std::cout << std::endl;
  std::string output_path;
  std::string image_list_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddGlobalMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(output_path)) {
    std::cerr << "ERROR: `output_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  // BINGO-TODO: I think there is a bug here
  // if (!image_list_path.empty()) {
  //   const auto image_names = ReadTextFileLines(image_list_path);
  //   options.mapper->image_names =
  //       std::unordered_set<std::string>(image_names.begin(), image_names.end());
  // }

  ReconstructionManager reconstruction_manager;
  GlobalMapperController mapper(options.global_mapper.get(), *options.image_path,
                                *options.database_path,
                                &reconstruction_manager);
  mapper.Start();
  mapper.Wait();

  for (size_t i = 0; i < reconstruction_manager.Size(); ++i) {
    const std::string reconstruction_path = JoinPaths(
        output_path, std::to_string(i));
    const auto& reconstruction =
        reconstruction_manager.Get(i);
    CreateDirIfNotExists(reconstruction_path);
    reconstruction.Write(reconstruction_path);
    options.Write(JoinPaths(reconstruction_path, "project.ini"));
  }

  return EXIT_SUCCESS;
}

typedef std::function<int(int, char**)> command_func_t;

int ShowHelp(
    const std::vector<std::pair<std::string, command_func_t>>& commands) {
  std::cout << StringPrintf("GCOLMAP -- Global Structure-from-Motion with COLMAP database\n")<< std::endl;
  std::cout << "Usage:" << std::endl;
  std::cout << "  gcolmap help [ -h, --help ]" << std::endl;
  std::cout << "  gcolmap global_mapper --image_path IMAGES --database_path DATABASE "
               "--output_path MODEL"
            << std::endl;
  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::vector<std::pair<std::string, command_func_t>> commands;
  commands.emplace_back("global_mapper", &RunGlobalMapper);

  if (argc == 1) {
    return ShowHelp(commands);
  }

  const std::string command = argv[1];
  if (command == "help" || command == "-h" || command == "--help") {
    return ShowHelp(commands);
  } else {
    command_func_t matched_command_func = nullptr;
    for (const auto& command_func : commands) {
      if (command == command_func.first) {
        matched_command_func = command_func.second;
        break;
      }
    }
    if (matched_command_func == nullptr) {
      std::cerr << StringPrintf(
                       "ERROR: Command `%s` not recognized. To list the "
                       "available commands, run `gcolmap help`.",
                       command.c_str())
                << std::endl;
      return EXIT_FAILURE;
    } else {
      int command_argc = argc - 1;
      char** command_argv = &argv[1];
      command_argv[0] = argv[0];
      return matched_command_func(command_argc, command_argv);
    }
  }

  return ShowHelp(commands);
}
