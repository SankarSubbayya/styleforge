#!/usr/bin/env bash
# Generate a 45s synthetic test clip (video pattern + tone + counter) for pipeline smoke tests.
set -euo pipefail
mkdir -p data/clips
ffmpeg -y -v error \
  -f lavfi -i "testsrc2=duration=45:size=640x360:rate=24" \
  -f lavfi -i "sine=frequency=440:duration=45" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac \
  data/clips/test_synthetic.mp4
echo "wrote data/clips/test_synthetic.mp4"
