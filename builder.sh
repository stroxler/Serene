#! /bin/bash

command=$1


export CCC_CC=clang-10
export CCC_CXX=clang++-10
export CC=clang-10
export CXX=clang++-10

ROOT_DIR=`pwd`
BUILD_DIR=$ROOT_DIR/build

scanbuild=scan-build-10

function build() {
    pushd $BUILD_DIR
    cmake $ROOT_DIR
    make -j 4
    popd
}

function clean() {
    rm -rf $BUILD_DIR
}

function run() {
    pushd $BUILD_DIR
    $BUILD_DIR/bin/serene "$@"
    popd
}

case "$command" in
    "fresh-build")
        clean
        mkdir -p $BUILD_DIR
        build
        ;;
    "build")
        build
        ;;
    "run")
        run "$@"
        ;;

    "scan-build")
        clean
        mkdir -p $BUILD_DIR
        pushd $BUILD_DIR
        exec $scanbuild cmake $ROOT_DIR
        exec $scanbuild scan-build make -j 4
        popd
        ;;
    "clean")
        rm -rf $BUILD_DIR
        ;;
    *)
        echo "Commands: "
        echo "fresh-build - Build Serene from scratch"
        echo "build - reCompiles the project using the already exist cmake configuration"
        echo "run - Runs the serene executable"
        echo "scan-build - Compiles serene with static analyzer"
        echo "clean - :D"
        ;;
esac
