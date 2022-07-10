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
# This file contains some helper functions to build the OCI images for
# development and production purposes of Serene.
#
# We use `Buildkit` to build the images and `podman` to run them.
#
# NOTE:
# REGISTERY comes frome the `.env` file in the root of the project
#
# NOTE:
# If you run into an issue like this with podman:
#
# WARN[0000] Error running newgidmap: exit status 1: newgidmap: gid range [1-1) -> [0-0)
# not allowed
# WARN[0000] Falling back to single mapping
#
# with podman or buildah try the following commands as root:
#
# usermod --add-subgids 1001000000-1001999999 YOUR_USER_NAME
# usermod --add-subgids 1001000000-1001999999 YOUR_USER_NAME
#
# or set the subuid and guid manually in `/etc/subuid` and `/etc/subgid`


set -e

export BUILDER_NAME="multiarch"
export PLATFORMS=('amd64' 'arm64')

function setup_builder() {
    info "Creating the builder container"
    sudo podman run --privileged --name "$BUILDER_NAME" \
         docker.io/multiarch/qemu-user-static --reset -p yes
}

function cleanup_builder() {
    info "Stopping the builder"
    sudo podman stop "$BUILDER_NAME"
    sudo podman rm "$BUILDER_NAME"
}

function create_manifest() {
    local manifest="$1"

    info "Remove the manifest if it exists"
    buildah manifest rm "${manifest}" || true
    info "Creating the manifest"
    buildah manifest create "${manifest}" || warn "Manifest exists"
}

function build_container_image() {
    local IMAGE_NAME="$1"
    local LLVM_VERSION="$2"
    local DOCKERFILE="$3"
    local ROOT="$4"
    local MANIFEST
    local IMAGE

    MANIFEST="serene/$1:${LLVM_VERSION}-$(git describe)"
    IMAGE="$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)"

    create_manifest "${MANIFEST}"

    for ARCH in "${PLATFORMS[@]}"; do
        info "Building the multiarch '$IMAGE_NAME' images for:"
        info "VERSION: $LLVM_VERSION | ARCH: $ARCH"

        buildah build \
                --jobs="$(nproc)" \
                --arch "$ARCH" \
                --layers \
                --manifest "${MANIFEST}" \
                -f "$DOCKERFILE" \
                -t "$IMAGE" \
                --build-arg VERSION="$LLVM_VERSION" \
                "$ROOT"

        # info "Tagging the image '$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)' as latest"
        # buildah tag \
        #    "$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)" \
        #    "$REGISTRY/$IMAGE_NAME:latest"
    done


    info "inspect ${MANIFEST}"
    buildah manifest inspect "${MANIFEST}"
    info "first push docker://$IMAGE"
    buildah manifest push --all "${MANIFEST}" \
            "docker://$IMAGE"
    info "second push  docker://$REGISTRY/$IMAGE_NAME:latest"
    buildah manifest push --all "${MANIFEST}" \
            "docker://$REGISTRY/$IMAGE_NAME:latest"
}

function build_llvm() {
    local LLVM_VERSION="$1"
    local ROOT="$2"

    build_container_image "llvm" "$LLVM_VERSION" "$ROOT/resources/docker/llvm/Dockerfile" "$ROOT"
}

function build_ci() {
    local LLVM_VERSION="$1"
    local ROOT="$2"

    build_container_image "ci" "$LLVM_VERSION" "$ROOT/resources/docker/llvm/Dockerfile.ci" "$ROOT"
}

function push_images() {
    local image="$1"

    local manifest
    manifest="serene/$image"

    buildah manifest push "${manifest}" \
            --creds "$SERENE_REGISTERY_USER:$SERENE_REGISTERY_PASS" \
            --all
}








function setup_builder1() {
    if ! docker buildx inspect --builder "$BUILDER_NAME"; then
        info "Creating the builder '$BUILDER_NAME'"
        docker buildx  create --driver docker-container \
               --name "$BUILDER_NAME" \
               --platform "linux/amd64,linux/arm64" \
               --use \
               --bootstrap
    else
        info "The builder '$BUILDER_NAME' already exists."
    fi
}

# Params:
# 2nd: Image tag to use it should be the major number of llvm version
# 3rd: Project root
function build_llvm_multiarch() {

    setup_builder

    local IMAGE_NAME="llvm"
    local LLVM_VERSION="$1"
    local ROOT="$2"

    info "Building the multiarch llvm images for:"
    info "VERSION: $LLVM_VERSION | Platforms: $PLATFORMS"
    docker buildx build --platform  "$PLATFORMS" \
           --builder "$BUILDER_NAME" --push \
           -f "$ROOT/resources/docker/llvm/Dockerfile" \
           -t "$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)" \
           --build-arg VERSION="$LLVM_VERSION" \
           "$ROOT"

    info "Tagging the image '$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)' as latest"
    docker tag \
           "$REGISTRY/$IMAGE_NAME:${LLVM_VERSION}-$(git describe)" \
           "$REGISTRY/$IMAGE_NAME:latest"
}


function build_ci_image() {
    setup_builder

    local LLVM_VERSION="$1"
    local ROOT="$2"
    local IMAGE

    IMAGE="$REGISTRY/ci:${LLVM_VERSION}-$(git describe)"

    info "Building the CI images"
    docker buildx build --platform  "linux/arm64" \
           --builder "$BUILDER_NAME" --load \
           -f "$ROOT/resources/docker/llvm/Dockerfile.ci" \
           -t "$IMAGE" \
           --build-arg VERSION="$2" \
           "$ROOT"

    info "Finished building '$IMAGE'"
    info "Tagging the image '$IMAGE' as latest"
    docker tag \
           "$IMAGE" \
           "$REGISTRY/ci:latest"

}


function push_images() {
    local LLVM_VERSION="$1"
    local ROOT="$2"

    info "Loging into registry"
    docker login "$REGISTRY" -u "$SERENE_REGISTERY_USER" -p "$SERENE_REGISTERY_PASS"

    info "Push the LLVM image"
    push "$REGISTRY/llvm:${LLVM_VERSION}-$(git describe)"
    push "$REGISTRY/llvm:latest"

    info "Push the CI image"
    push "$REGISTRY/ci:${LLVM_VERSION}-$(git describe)"
    push "$REGISTRY/ci:latest"
}
