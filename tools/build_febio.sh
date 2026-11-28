#!/usr/bin/env bash
# Build FEBio from source for reference runs. No MKL and no optional
# libraries, so it needs only git, cmake, make and a C++ compiler. About 40
# minutes on two cores. The result is a provisional build, not an official
# release: FEBio then uses its skyline solver, and results should be
# regenerated on a release build before they are relied on.
#
#   tools/build_febio.sh [work directory]        default: ~/febio
#   FEBIO_COMMIT=<hash> tools/build_febio.sh     another commit
#   FEBIO_STATIC_STDLIBS=OFF tools/build_febio.sh
#       where the static C++ runtime is not installed (the link then fails
#       with "cannot find -lstdc++"; on RHEL the package is libstdc++-static).
#       FEBio's default is ON. This changes how the runtime is linked only.
#
# Prints the path of the febio4 binary. Use it as
#   FEBIO_BIN=<path> python references/small_problems/make_references.py ...
set -euo pipefail

# The commit the references added with the run path template were made at
# (develop, November 25, 2026). Earlier references in this repository name
# their own commit in their README.
COMMIT="${FEBIO_COMMIT:-e4df3f8ea6d1cc3e2128d342778859000bbf5609}"
STATIC_STDLIBS="${FEBIO_STATIC_STDLIBS:-ON}"
WORK="${1:-$HOME/febio}"
SRC="$WORK/FEBio"
BUILD="$WORK/build"

mkdir -p "$WORK"
if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/febiosoftware/FEBio.git
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC" checkout -q FETCH_HEAD

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MKL=OFF -DUSE_HYPRE=OFF -DUSE_MMG=OFF -DUSE_LEVMAR=OFF \
    -DUSE_FFTW=OFF -DUSE_ZLIB=OFF -DUSE_NLOPT=OFF -DUSE_SUPERLU_MT=OFF \
    -DUSE_STATIC_STDLIBS="$STATIC_STDLIBS" \
    -DCMAKE_EXE_LINKER_FLAGS=-fopenmp -DCMAKE_SHARED_LINKER_FLAGS=-fopenmp \
    > "$WORK/cmake.log"
make -C "$BUILD" -j"$(nproc)" febio4 > "$WORK/make.log" 2>&1

BIN="$(find "$BUILD" -type f -name febio4 -perm -u+x | head -1)"
"$BIN" -info 2>/dev/null | head -5 || true
echo "$BIN"
