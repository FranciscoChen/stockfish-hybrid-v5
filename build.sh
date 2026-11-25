#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

eval_file=stockfish/src/nn-c9e480e92425.nnue
eval_checksum=c9e480e92425f887df6ab9b05c6180fbd762b07879df9d9845f1aa75440caa73
policy_file=weights/v5-7a702165ba92.policy
policy_checksum=7a702165ba92072006f99708a16aeb60508130d9044412a17a29ad526b04008e

verify() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}
if ! verify "$policy_checksum" "$policy_file"; then
    echo "Policy checksum failed for $policy_file" >&2
    exit 1
fi
if ! verify "$eval_checksum" "$eval_file"; then
    temp_file=$(mktemp "$eval_file.XXXXXX")
    trap 'rm -f -- "$temp_file"' EXIT
    gzip -dc -- "weights/${eval_file##*/}.gz" > "$temp_file"
    if ! verify "$eval_checksum" "$temp_file"; then
        echo "Evaluation checksum failed for $eval_file" >&2
        exit 1
    fi
    mv -- "$temp_file" "$eval_file"
fi
make -C stockfish/src objclean
make -C stockfish/src -j"${JOBS:-1}" build ARCH="${ARCH:-x86-64}" \
    EXTRACXXFLAGS=-DCHESSIL_EMBED_POLICY
