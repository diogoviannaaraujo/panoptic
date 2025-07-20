#!/usr/bin/env bash
# build-gstreamer-monorepo.sh  ────────────────────────────────────────────────
# Builds ONNX Runtime v1.16.3 (CPU or CUDA) *then* the GStreamer monorepo
# with motioncells + onnxinference enabled.
set -euo pipefail

### 0.  What we’re building ----------------------------------------------------
GST_VER="${GST_VER:-$(gst-launch-1.0 --version 2>/dev/null | awk '/GStreamer/{print $2}' | cut -d. -f1,2)}"
GST_TAG="${GST_TAG:-${GST_VER:-1.24}}"     # fallback to 1.24.x if gst not installed
ONNX_TAG="v1.16.3"
SRC_DIR="${SRC_DIR:-$HOME/src}"
BUILD_DIR="${BUILD_DIR:-$HOME/build}"

echo "▶ GStreamer tag:  $GST_TAG  (monorepo)"
echo "▶ ONNX Runtime:   $ONNX_TAG   (CUDA=${USE_CUDA:-0})"

### 1.  Build deps ------------------------------------------------------------
sudo apt update
sudo apt install -y --no-install-recommends \
  build-essential git cmake ninja-build meson pkg-config \
  libglib2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libopencv-dev libeigen3-dev flex bison

### 2.  Build ONNX Runtime -----------------------------------------------------
mkdir -p "$SRC_DIR" "$BUILD_DIR"
cd "$SRC_DIR"

if [[ ! -d onnxruntime ]]; then
  git clone --recursive https://github.com/microsoft/onnxruntime.git
fi
cd onnxruntime
git fetch --tags
git checkout -B "$ONNX_TAG" "refs/tags/$ONNX_TAG"

mkdir -p "$BUILD_DIR/onnxruntime" && cd "$BUILD_DIR/onnxruntime"

onnx_flags=(
  -Donnxruntime_BUILD_SHARED_LIB=ON
  -DBUILD_TESTING=OFF -Donnxruntime_BUILD_UNIT_TESTS=OFF
  -Donnxruntime_USE_PREINSTALLED_EIGEN=ON -Deigen_SOURCE_PATH=/usr/include/eigen3
)
if [[ "${USE_CUDA:-0}" == 1 ]]; then
  onnx_flags+=(
    -Donnxruntime_USE_CUDA=ON
    -Donnxruntime_CUDA_HOME=/usr/local/cuda
    -Donnxruntime_CUDNN_HOME=/usr/local/cuda
    -DCMAKE_CUDA_ARCHITECTURES=native
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
  )
fi

cmake "${onnx_flags[@]}" "$SRC_DIR/onnxruntime/cmake"
make -j"$(nproc)"
sudo make install                # /usr/local/{lib,include}

### 3.  Clone the monorepo -----------------------------------------------------
cd "$SRC_DIR"
if [[ ! -d gstreamer ]]; then
  git clone --depth 1 --branch "$GST_TAG" \
    https://gitlab.freedesktop.org/gstreamer/gstreamer.git
fi
cd gstreamer

### 4.  Configure Meson --------------------------------------------------------
#   Namespaced options are   -D<subproject>:<option>=value
#   We only enable the bits we need to cut compile time.
meson setup build \
  --prefix=/usr/local --buildtype=release \
  -Dbad=enabled

### 5.  Build & install --------------------------------------------------------
meson compile -C build
sudo meson install -C build
sudo ldconfig

### 6.  Smoke-test -------------------------------------------------------------
echo
echo "🩺  gst-inspect-1.0 checks"
gst-inspect-1.0 motioncells     >/dev/null && echo "  ✓ motioncells found"
gst-inspect-1.0 onnxinference   >/dev/null && echo "  ✓ onnxinference found"
echo "Done – new plugins are in /usr/local/lib/gstreamer-1.0"