#!/bin/sh
# Old name: the project is Saguaro now.  ./maya.sh runs ./saguaro.sh, kept so old instructions keep working.
cd "$(dirname "$0")" || exit 1
exec ./saguaro.sh "$@"
