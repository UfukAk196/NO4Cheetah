#!/bin/bash -l
set -euo pipefail

source /software/spack/activate_spack.sh
spack load opencode@2.2.5

cd  ~/NO4Cheetah/Pipeline

opencode