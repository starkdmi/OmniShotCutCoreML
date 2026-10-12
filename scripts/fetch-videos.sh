#!/bin/sh
# Downloads the reference videos into a directory (default: videos/) and
# checks them against the hashes the reference was made from. A download whose
# hash differs is deleted. All three are Blender Foundation open movies,
# CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/), (c) Blender
# Foundation: Sintel (https://durian.blender.org), Big Buck Bunny
# (https://peach.blender.org) and Tears of Steel (https://mango.blender.org).
set -eu
directory="${1:-videos}"
mkdir -p "$directory"
cd "$directory"

fetch() {
    name="$1" url="$2" sha256="$3"
    if [ ! -f "$name" ] || ! echo "$sha256  $name" | shasum -a 256 -c -s; then
        echo "downloading $name"
        curl -fL --progress-bar -o "$name" "$url"
    fi
    if ! echo "$sha256  $name" | shasum -a 256 -c; then
        rm -f "$name"
        exit 1
    fi
}

# Sintel trailer: fades and titles.
fetch sintel_trailer-480p.mp4 \
    https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 \
    b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254
# Big Buck Bunny trailer: hard cuts; 853 pixels wide, tagged BT.709.
fetch trailer_480p.mov \
    https://download.blender.org/peach/trailer/trailer_480p.mov \
    36801b74638c12be9aa587e93cd18edfc9bc51a1c089ab2a19ee42beed9f497d
# Tears of Steel, the whole film: 12 minutes (17620 frames), 196 windows of
# 100 frames overlapping by 10. 372 MB (388 MB for all three).
fetch tears_of_steel_720p.mov \
    https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov \
    efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f
