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

#include "leela_policy.h"
#include <cmath>
#if defined(__linux__)
#include <dlfcn.h>
#endif
namespace Stockfish {
LeelaPolicy::~LeelaPolicy() { close(); }
void LeelaPolicy::close() {
    if (network && destroy) destroy(network);
    network = nullptr;
#if defined(__linux__)
    if (libraryHandle) dlclose(libraryHandle);
#endif
    libraryHandle = nullptr;
    destroy = nullptr;
    infer = nullptr;
    configuration.clear();
}
bool LeelaPolicy::evaluate(const std::string& library, const std::string& weights,
                          const std::string& backend, const std::string& fen,
                          const std::vector<std::string>& history,
                          const std::vector<std::string>& moves,
                          std::vector<float>& probabilities, std::string& error) {
#if defined(__linux__)
    const std::string config = library + '\n' + weights + '\n' + backend;
    if (!network || config != configuration) {
        close();
        libraryHandle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!libraryHandle) { error = dlerror(); return false; }
        auto create = reinterpret_cast<Create>(dlsym(libraryHandle, "sf_leela_create_v1"));
        destroy = reinterpret_cast<Destroy>(dlsym(libraryHandle, "sf_leela_destroy_v1"));
        infer = reinterpret_cast<Evaluate>(dlsym(libraryHandle, "sf_leela_policy_v1"));
        if (!create || !destroy || !infer) {
            error = "Leela policy library has an incompatible API"; close(); return false;
        }
        char message[1024]{};
        network = create(weights.c_str(), backend.c_str(), message, sizeof(message));
        if (!network) { error = message; close(); return false; }
        configuration = config;
    }
    std::string played;
    for (const auto& move : history) played += move + ' ';
    std::vector<const char*> legal;
    for (const auto& move : moves) legal.push_back(move.c_str());
    probabilities.resize(moves.size());
    char message[1024]{};
    if (!infer(network, fen.c_str(), played.c_str(), legal.data(), unsigned(legal.size()),
               probabilities.data(), message, sizeof(message))) {
        error = message; probabilities.clear(); return false;
    }
    float sum = 0;
    for (float p : probabilities) {
        if (!std::isfinite(p) || p < 0) { error = "Invalid policy output"; return false; }
        sum += p;
    }
    if (!std::isfinite(sum) || sum <= 0) { error = "Empty policy output"; return false; }
    return true;
#else
    (void)library; (void)weights; (void)backend; (void)fen; (void)history;
    (void)moves; (void)probabilities;
    error = "Native Leela policy currently requires Linux";
    return false;
#endif
}
}
