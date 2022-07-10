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
# Bunch of helper function to create a container like environment to develop
# Serene on GNU/Linux without going through the headache of compiling LLVM
# 5 times a day :D

set -e


function as_root() {
    local rootfs="$1"
    sudo unshare \
         -w "/serene" \
         --uts \
         --ipc \
         --pid \
         --fork \
         --kill-child \
         --cgroup \
         --mount \
         --mount-proc \
         --root="$rootfs" \
         "${@:2}"
}


function rootless() {
    local rootfs="$1"
    unshare \
        -w "/serene" \
        --uts \
        -c \
        --ipc \
        --pid \
        --fork \
        --kill-child \
        --cgroup \
        --mount \
        --mount-proc \
        --root="$rootfs" \
        "${@:2}"
}

function download_devfs() {
    local repo="$1"
    local target="$2"
    info "Downloading the tarball from '$repo'"
    wget "$repo/fs.latest.tar.xz" -O "$target"
}


function extract_devfs() {
    local tarball="$1"
    local to="$2"

    info "Extracting the tarball..."
    mkdir -p "$to"
    tar Jxf "$tarball" -C "$to"

    info "Create the 'serene' dir at the root"
    mkdir -p "$to/serene"
}

function mount_serene {
    local rootfs="$1"
    local project_root="$2"
    local serene_dir

    serene_dir="$rootfs/serene"

    mkdir -p "$serene_dir"

    info "Mounting Serene's dir into '/serene' in read-only mode"
    mountpoint -q "$serene_dir" || sudo mount --rbind -o ro "$project_root" "$serene_dir"
}

function mount_trees() {
    local rootfs="$1"
    local project_root="$2"

    mount_serene "$rootfs" "$project_root"

    info "Mounting the 'tmpfs' at '$rootfs/tmp'"
    mkdir -p "$rootfs/tmp"
    mountpoint -q "$rootfs/tmp" || sudo mount -t tmpfs tmpfs "$rootfs/tmp"

    info "Mounting 'dev' at '$rootfs/dev'"
    mkdir -p "$rootfs/dev"
    mountpoint -q "$rootfs/dev" || sudo mount --bind /dev "$rootfs/dev"
}


function unmount_trees() {
    local rootfs="$1"

    info "Unmounting the 'serene' from '$rootfs/serene'"
    sudo umount "$rootfs/serene" &> /dev/null || true

    info "Unmounting the 'tmpfs' from '$rootfs/tmp'"
    sudo umount "$rootfs/tmp" &> /dev/null || true

    info "Unmounting 'dev' from '$rootfs/dev'"
    sudo umount "$rootfs/dev" &> /dev/null || true
}

function init_devfs {
    local rootfs="$1"
    local project_root="$2"
    local create_group
    local create_user

    create_group="groupadd -f -g$(id -g) $(whoami)"
    create_user="adduser -q --disabled-password --gecos '' --uid $(id -u) --gid $(id -g) $(whoami) || true"

    mkdir -p "$rootfs/proc"
    mount_trees "$rootfs" "$project_root"

    info "Creating group '$(whoami)' with ID '$(id -g)'..."
    as_root "$rootfs" bash --login -c "$create_group"

    info "Creating user '$(whoami)' with ID '$(id -u)'..."
    as_root "$rootfs" bash --login -c "$create_user"

    info "Make '$(whoami)' a sudoer"
    as_root "$rootfs" bash --login -c "adduser $(whoami) sudo || true"
}

function create_debian_rootfs() {
    local to="$1"

    info "Pulling the debian docker image"
    docker pull debian:sid-slim

    info "Spinning up the container"
    docker stop devfs &> /dev/null || true
    docker rm devfs &> /dev/null || true
    docker run --name devfs -d debian:sid-slim
    sleep 2

    info "Exporting the rootfs to '$to/rootfs.tar'"
    docker export -o "$to/rootfs.tar" devfs

    info "Tearing down the container"
    docker stop devfs &> /dev/null
    docker rm devfs &> /dev/null
}

function create_and_initialize_devfs_image() {
    local to="$1"
    local project_root="$2"
    local llvm_version="$3"
    local rootfs
    local version

    version=$(git describe)
    rootfs="$to/rootfs/"

    if [ ! -f "$to/rootfs.tar" ]; then
        info "Creating the rootfs tar bar"
        create_debian_rootfs "$to"
    fi

    mkdir -p "$rootfs"

    if [ ! -f "$to/rootfs/etc/shadow" ]; then
        info "Extracting the tarball"
        tar xf "$to/rootfs.tar" -C "$rootfs"
    fi

    mount_trees "$rootfs" "$project_root"

    #as_root "$rootfs" bash

    as_root "$rootfs" bash -c "echo '$llvm_version' > /etc/llvm_version"
    as_root "$rootfs" bash -c "echo 'export LANG=C.UTF-8' >> /etc/bash.bashrc"
    as_root "$rootfs" bash -c "echo 'export LANG=C.UTF-8' >> /etc/profile"
    as_root "$rootfs" bash /serene/scripts/devfs_container_setup.sh

    unmount_trees "$rootfs"

    info "Creating the tarball (It will take a few minutes)"
    sudo tar c -C "$rootfs" "." | lzma -c -9 -T "$(nproc)" > "$to/fs.$version.tar.xz"

    info "Removing the exporter tarball"
    rm -fv "$to/rootfs.tar"
}

function sync_devfs_image() {
    local where="$1"
    local version

    version=$(git describe)

    yes_or_no "Upload images '$where/fs.$version.tar.xz'?" && \
    info "Uploading image '$where/fs.$version.tar.xz'" && \
    rsync -vlc --progress \
          "$where/fs.$version.tar.xz" \
          core.lxsameer.com:/home/www/public/dl.serene/devfs/
}

function mark_devfs_image_as_latest() {
    local to="$1"
    local version
    local remote_cmd

    version=$(git describe)
    info "Marking images 'fs.$version.tar.xz' as latest"
    remote_cmd="rm -f /home/www/public/dl.serene/devfs/fs.latest.tar.xz && ln -s /home/www/public/dl.serene/devfs/fs.$version.tar.xz /home/www/public/dl.serene/devfs/fs.latest.tar.xz"

    # shellcheck disable=SC2029
    ssh core.lxsameer.com "$remote_cmd"
}

function unmount_and_destroy_devfs() {
    local rootfs="$1"

    unmount_trees "$rootfs"

    yes_or_no "Is it correct? 'sudo rm -rfv $rootfs'?" && sudo rm -rfv "$rootfs"
}
