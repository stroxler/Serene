#!/bin/sh
USER=www
HOST=core.lxsameer.com
DIR=/home/www/public/serene/
SITEDIR=$(cd "$(dirname "$0")/.." >/dev/null 2>&1 ; pwd -P)

hugo && rsync -avz --delete ${SITEDIR}/public/ ${USER}@${HOST}:${DIR}

exit 0
