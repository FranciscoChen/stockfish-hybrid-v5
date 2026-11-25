/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LEELA_POLICY_H_INCLUDED
#define LEELA_POLICY_H_INCLUDED
#include <string>
#include <vector>

namespace Stockfish {
// Native, optional backend with a C ABI. Each worker owns its inference state.
class LeelaPolicy {
 public:
    LeelaPolicy() = default;
    LeelaPolicy(const LeelaPolicy&) = delete;
    LeelaPolicy& operator=(const LeelaPolicy&) = delete;
    ~LeelaPolicy();
    bool evaluate(const std::string& library, const std::string& weights,
                  const std::string& backend, const std::string& fen,
                  const std::vector<std::string>& history,
                  const std::vector<std::string>& moves, std::vector<float>& probabilities,
                  std::string& error);
 private:
    void close();
    void* libraryHandle = nullptr;
    void* network = nullptr;
    using Create = void* (*)(const char*, const char*, char*, unsigned);
    using Destroy = void (*)(void*);
    using Evaluate = int (*)(void*, const char*, const char*, const char* const*, unsigned,
                            float*, char*, unsigned);
    Destroy destroy = nullptr;
    Evaluate infer = nullptr;
    std::string configuration;
};
}
#endif
