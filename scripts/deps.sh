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

set -e

function get_package() {
    local pkg_name
    local version
    local repository
    local force
    local pkg_path

    pkg_name="$1"
    version="$2"
    repository="$3"
    pkg_path=$(realpath "$BUILDER_CACHE_DIR/packages/$pkg_name/$version.zstd")
    force="false"

    # for reuse we need to set to OPTIND=1 explicitly
    #
    # while getopts ":n:v:r:f" opt "$@{*}"; do
    #     case "$opt" in
    #         n)
    #             echo "$OPTARG"
    #             pkg_name="$OPTARG"
    #             ;;
    #         v)
    #             version="$OPTARG"
    #             ;;
    #         r)
    #             repository="$OPTARG"
    #             ;;
    #         f)
    #             force="true"
    #             ;;
    #         \?)
    #             echo "here"
    #             ;;
    #     esac
    # done

    info "Looking up package '$pkg_name' version '$version'..."


    if [ -f "$pkg_path" ] && [ "$force" = "false" ]; then
        info "The package exists: $pkg_path"
        info "Skipping..."
        #info "Use '-f' to download it anyway."
        return
    fi

    mkdir -p "$BUILDER_CACHE_DIR/packages/$pkg_name"
    curl "$repository/api/packages/Serene/generic/$pkg_name/$version/$pkg_name.$version.zstd" \
         --progress-bar -o "$BUILDER_CACHE_DIR/packages/$pkg_name/${version}.zstd"
    info "Done"
}

function unpack_package() {
    local pkg_name
    local version
    local pkg_path
    local unpack_path

    pkg_name="$1"
    version="$2"

    pkg_path=$(realpath "$BUILDER_CACHE_DIR/packages/$pkg_name/$version.zstd")
    unpack_path=$(realpath "$BUILDER_CACHE_DIR/packages/$pkg_name/$version")

    info "Unpacking package '$pkg_name' version '$version'..."
    if [[ -f "$unpack_path" ]]; then
        info "The package exists: $unpack_path"
        info "Skipping..."
        return
    fi

    mkdir -p "$unpack_path"
    tar -I 'zstd --ultra -22' -C "$unpack_path" -xf "$pkg_path" --strip-components=1

}
