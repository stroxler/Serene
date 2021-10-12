/* -*- C++ -*-
 * Serene Programming Language
 *
 * Copyright (c) 2019-2021 Sameer Rahmani <lxsameer@gnu.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serene/serene.h"

#include <linenoise.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

using namespace serene;
namespace cl = llvm::cl;

static std::string banner =
    llvm::formatv("\nSerene Compiler Version {0}"
                  "\nCopyright (C) 2019-2021 "
                  "Sameer Rahmani <lxsameer@gnu.org>\n"
                  "Serene comes with ABSOLUTELY NO WARRANTY;\n"
                  "This is free software, and you are welcome\n"
                  "to redistribute it under certain conditions; \n"
                  "for details take a look at the LICENSE file.\n",
                  SERENE_VERSION);

static std::string art = "\n";

static cl::opt<std::string>
    historyFile("h", cl::desc("The absolute path to the history file to use."),
                cl::value_desc("filename"), cl::init("~/.serene-repl.history"));

int main(int argc, char *argv[]) {
  initCompiler();
  registerSereneCLOptions();

  cl::ParseCommandLineOptions(argc, argv, banner);

  llvm::outs() << banner << art;

  auto ctx    = makeSereneContext();
  auto userNS = makeNamespace(*ctx, "user", llvm::None);

  applySereneCLOptions(*ctx);

  // Enable the multi-line mode
  linenoise::SetMultiLine(true);

  // Set max length of the history
  linenoise::SetHistoryMaxLen(4);

  // Load history
  linenoise::LoadHistory(historyFile.c_str());

  while (true) {
    // Read line
    std::string line;
    auto quit = linenoise::Readline("user> ", line);

    if (quit) {
      break;
    }

    llvm::outs() << "echo: '" << line << "'"
                 << "\n";

    // Add text to history
    linenoise::AddHistory(line.c_str());
  }

  // Save history
  linenoise::SaveHistory(historyFile.c_str());
}
