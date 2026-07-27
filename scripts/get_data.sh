#!/usr/bin/env bash
# Download the fifteen UCI datasets. ~2.5 MB total.
set -euo pipefail
cd "$(dirname "$0")/../data" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data"; cd "$(dirname "$0")/../data"; }
for spec in "52 ionosphere" "94 spambase" "109 wine" "174 parkinsons" \
            "267 banknote+authentication" "42 glass+identification" \
            "212 vertebral+column" "176 blood+transfusion+service+center" \
            "372 htru2" "236 seeds" "161 mammographic+mass" "850 raisin" \
            "254 qsar+biodegradation" "329 diabetic+retinopathy+debrecen" "33 dermatology"; do
  id=${spec%% *}; nm=${spec#* }
  [ -d "d${id}" ] && continue
  curl -sfL -o "d${id}.zip" "https://archive.ics.uci.edu/static/public/${id}/${nm}.zip" \
    && unzip -oq "d${id}.zip" -d "d${id}" && echo "ok $id" || echo "FAILED $id"
done
