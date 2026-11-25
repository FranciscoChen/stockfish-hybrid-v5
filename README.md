# ChessiL Stockfish Hybrid V5

Standalone source snapshot of ChessiL's Stockfish 19 development hybrid, based
on Stockfish revision `2ffd877d`. It includes the full evaluation NNUE and the
frozen V5 policy network used in the ChessiL Hybrid V5 release.

## Build on Linux

Install Bash, GNU Make, a C++17-capable GCC toolchain, gzip, and GNU coreutils.
From the repository root run:

```sh
./build.sh
./stockfish/src/stockfish
```

The build checks the network checksums, unpacks the evaluation weights, and
embeds both networks in the executable. The build works offline and the
executable can run from any working directory. The default target is
x86-64 with SSE2, using one build job. For a compatible CPU, use
`ARCH=x86-64-bmi2 JOBS=2 ./build.sh` to select BMI2 and two build jobs.

| Network | Repository file | UCI option |
| --- | --- | --- |
| Evaluation NNUE | `weights/nn-c9e480e92425.nnue.gz` | `EvalFile` |
| V5 move policy | `weights/v5-7a702165ba92.policy` | `CompactPolicyFile` |

The engine defaults to `PolicySearch=true`, `PolicySearchReductions=false`,
and `CompactPolicyFile=<embedded>`. To use replacement weights, set the
corresponding UCI option to the path of an uncompressed network file. Set
`PolicySearch=false` to disable the search policy.

## Browser build

The browser adapter and applied patch are in `browser/`. With Emscripten
6.0.9 active, run:

```sh
bash browser/build.sh stockfish/src output simd
cp browser/worker.js weights.json output/
cp -r weights output/
```

Serve `output/` over HTTP and load `worker.js` from a browser Worker. The
browser build loads the two weight assets from `weights/` at startup.

## License

Stockfish and the hybrid source are distributed under the GNU General Public
License, version 3 or later. See [license](stockfish/Copying.txt) and
[Stockfish contributors](stockfish/AUTHORS).
