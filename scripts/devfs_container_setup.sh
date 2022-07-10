#! /bin/bash
# Serene Programming Language
#
# Copyright (c) 2019-2022 Sameer Rahmani <lxsameer@gnu.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# -----------------------------------------------------------------------------
# Commentary
# -----------------------------------------------------------------------------
# This file installs all the dependencies on the guest container during the
# devfs initialization.

set -e

# shellcheck source=/dev/null
source /serene/scripts/utils.sh


function install_llvm() {

    wget https://apt.llvm.org/llvm.sh -O /root/llvm.sh
    chmod +x llvm.sh

    /root/llvm.sh "${LLVM_VERSION}" all

    apt-get update --fix-missing
    apt-get install -y --no-install-recommends \
            mlir-"${LLVM_VERSION}"-tools \
            libmlir-"${LLVM_VERSION}"-dev \
            libmlir-"${LLVM_VERSION}" \
            libmlir-"${LLVM_VERSION}"-dbgsym \
            liblld-"${LLVM_VERSION}" \
            liblld-"${LLVM_VERSION}"-dev \
            clang-format-"${LLVM_VERSION}" \
            clang-tidy-"${LLVM_VERSION}"

    ln -s "$(which lld-"${LLVM_VERSION}")" /usr/bin/lld
    ln -s "$(which clang-"${LLVM_VERSION}")" /usr/bin/clang
    ln -s "$(which clang++-"${LLVM_VERSION}")" /usr/bin/clang++
    ln -s "$(which clang-format-"${LLVM_VERSION}")" /usr/bin/clang-format
    ln -s "$(which clang-tidy-"${LLVM_VERSION}")" /usr/bin/clang-tidy
    ln -s "$(which mlir-tblgen-"${LLVM_VERSION}")"  /usr/bin/mlir-tblgen

    MLIR_DIR="/usr/lib/llvm-${LLVM_VERSION}"
    CMAKE_PREFIX_PATH="/usr/lib/llvm-${LLVM_VERSION}"
    LD_LIBRARY_PATH="/usr/lib/llvm-${LLVM_VERSION}/lib/clang/${LLVM_VERSION}.0.0/lib/linux/"
    CC=/usr/bin/clang
    CXX=/usr/bin/clang++
}

function install_iwuy() {
    mkdir -p /opt/iwuy
    pushd /opt/iwuy
    git clone https://github.com/include-what-you-use/include-what-you-use.git --depth 1
    mkdir build
    pushd build
    cmake -G Ninja -DCMAKE_PREFIX_PATH="/usr/lib/llvm-${LLVM_VERSION}" ../include-what-you-use
    cmake --build .
    cmake -P cmake_install.cmake
    popd
    popd

    rm -rf /opt/iwuy
}

function install_boehm() {
    mkdir -p /opt/boehm
    pushd /opt/boehm
    git clone https://github.com/ivmai/bdwgc.git --depth 1 --branch v8.2.0
    mkdir build
    pushd build
    cmake -G Ninja -DBUILD_SHARED_LIBS=OFF -Denable_cplusplus=ON -Denable_threads=ON \
          -Denable_gcj_support=OFF -Dinstall_headers=ON \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../bdwgc
    cmake --build . --config Release
    cmake -P cmake_install.cmake
    popd
    popd

    rm -rf /opt/boehm
}

function main() {
    pushd "/root"

    apt-get update

    apt-get install --no-install-recommends -y \
            gnupg \
            cmake \
            ccache \
            git \
            ninja-build \
            binutils \
            lsb-release \
            wget \
            software-properties-common \
            zlib1g \
            cppcheck \
            sudo \
            shellcheck \
            zlib1g-dev

    # install_llvm
    # install_iwuy
    # install_boehm
    popd

    info "Enabling passwordless sudo"
    sed 's/%sudo.*/%sudo   ALL=(ALL) NOPASSWD:ALL/' -i /etc/sudoers

    apt-get autoremove -y
    apt-get clean
}

if [ ! -f "/etc/llvm_version" ]; then
    error "Can't find '/etc/llvm_version' on the container"
    exit 1
fi

export LANG=C.UTF-8
export LLVM_VERSION
export MLIR_DIR
export CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH
export CC
export CXX

LLVM_VERSION=$(cat /etc/llvm_version)

main
