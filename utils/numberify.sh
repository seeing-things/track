#!/bin/bash

# uses the number in the filename to determine ordering
# does NOT handle wraparound cases

 IN_DIR="$1"
OUT_DIR="$2"

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <in_dir> <out_dir>" >&2
	exit 1
fi

mkdir -p "$OUT_DIR"

FILES=($(find "$IN_DIR" -iname '*.jpg' | sort))

NUM=0
for FILE in "${FILES[@]}"; do
	ln -s "$FILE" "$OUT_DIR/$(printf "%04d" $NUM).jpg" # symlink
#	cp -a "$FILE" "$OUT_DIR/$(printf "%04d" $NUM).jpg" # copy
#	mv    "$FILE" "$OUT_DIR/$(printf "%04d" $NUM).jpg" # move
	(( ++NUM ))
done

exit 0
