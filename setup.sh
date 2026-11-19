#!/bin/sh
# Saguaro for Linux and macOS - ./setup.sh, another name for Saguaro's installer (./saguaro.sh): the first run sets
# everything up and starts the dashboard; later runs just start it.  ./setup.sh --help lists the options.
cd "$(dirname "$0")" || exit 1
exec ./saguaro.sh "$@"
