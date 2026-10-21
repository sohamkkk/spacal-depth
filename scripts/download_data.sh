#!/usr/bin/env bash
# Download + unpack evaluation data into $SPACAL_DATA_ROOT (default: ./data). ~8 GB total.
#   bash scripts/download_data.sh            # everything
#   bash scripts/download_data.sh nyu kitti  # subset
# Resumable (wget -c). Tarballs are deleted after a successful extract to save space.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${SPACAL_DATA_ROOT:-data}
TARGETS=${*:-nyu kitti diode}
MARIGOLD_RAW=https://raw.githubusercontent.com/prs-eth/Marigold/main/data_split/kitti_depth
MARIGOLD_DATA=https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/kitti

fetch() { wget -c --tries=5 --timeout=60 -O "$2" "$1"; }

for t in $TARGETS; do
  case "$t" in
    nyu)  # labelled subset (2.97 GB) + official 795/654 split
      mkdir -p "$D/nyu"
      fetch http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat "$D/nyu/nyu_depth_v2_labeled.mat"
      fetch http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat "$D/nyu/splits.mat"
      ;;
    kitti)  # Eigen test (652 w/ GT, 646 MB) + 800 frames from Eigen-train drives (756 MB)
      mkdir -p "$D/kitti/splits" "$D/kitti/eigen_test" "$D/kitti/val800"
      fetch "$MARIGOLD_RAW/eigen_test_files_with_gt.txt" "$D/kitti/splits/eigen_test_files_with_gt.txt"
      fetch "$MARIGOLD_RAW/eigen_val_from_train_800.txt" "$D/kitti/splits/eigen_val_from_train_800.txt"
      for pair in "kitti_eigen_split_test.tar:eigen_test" "kitti_sampled_val_800.tar:val800"; do
        tarball=${pair%%:*}; dest=${pair##*:}
        if [ -z "$(ls -A "$D/kitti/$dest")" ]; then
          fetch "$MARIGOLD_DATA/$tarball" "$D/kitti/$tarball"
          tar -xf "$D/kitti/$tarball" -C "$D/kitti/$dest" && rm "$D/kitti/$tarball"
        fi
      done
      ;;
    diode)  # official validation split (2.77 GB): 325 indoor + 446 outdoor
      mkdir -p "$D/diode"
      if [ ! -d "$D/diode/val" ]; then
        fetch http://diode-dataset.s3.amazonaws.com/val.tar.gz "$D/diode/val.tar.gz"
        tar -xzf "$D/diode/val.tar.gz" -C "$D/diode" && rm "$D/diode/val.tar.gz"
      fi
      ;;
    *) echo "unknown target $t (use nyu kitti diode)"; exit 1 ;;
  esac
done

echo "--- sanity counts"
[ -f "$D/nyu/nyu_depth_v2_labeled.mat" ] && ls -lh "$D/nyu"
[ -d "$D/kitti/eigen_test" ] && echo "kitti eigen_test pngs: $(find "$D/kitti/eigen_test" -name '*.png' | wc -l)"
[ -d "$D/kitti/val800" ] && echo "kitti val800 pngs: $(find "$D/kitti/val800" -name '*.png' | wc -l)"
[ -d "$D/diode/val" ] && echo "diode rgb: $(find "$D/diode/val" -name '*.png' | wc -l) (expect 771)"
du -sh "$D"
