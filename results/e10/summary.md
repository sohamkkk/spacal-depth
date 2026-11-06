## Inference (ms / image, GPU)

| dataset | model | t_pred_ms | t_flip_ms | t_grad_ms |
|---|---|---|---|---|
| diode_indoor | dav2-indoor-base | 55.34 |  |  |
| diode_indoor | dav2-indoor-large | 124.2 | 128.6 | 356.4 |
| diode_indoor | dav2-indoor-small | 32.89 |  |  |
| diode_indoor | unidepthv2-large | 60.56 | 61.72 |  |
| diode_outdoor | dav2-outdoor-base | 53.61 |  |  |
| diode_outdoor | dav2-outdoor-large | 123.3 | 128.2 | 356.2 |
| diode_outdoor | dav2-outdoor-small | 32.18 |  |  |
| diode_outdoor | unidepthv2-large | 58.76 | 61.13 |  |
| kitti | dav2-outdoor-base | 119.4 |  |  |
| kitti | dav2-outdoor-large | 321.2 | 326.9 | 1103 |
| kitti | dav2-outdoor-small | 61.04 |  |  |
| kitti | unidepthv2-large | 52.86 | 55.18 |  |
| nyu | dav2-indoor-base | 50.44 |  |  |
| nyu | dav2-indoor-large | 120.7 | 121.4 | 354 |
| nyu | dav2-indoor-small | 28.58 |  |  |
| nyu | unidepthv2-large | 45.92 | 47.02 |  |

## Calibration (ms, CPU, whole calibration set)

| dataset | model | calibrator | calib_ms |
|---|---|---|---|
| diode_indoor | dav2-indoor-large | K0 | 31.75 |
| diode_indoor | dav2-indoor-large | K1 | 719 |
| diode_indoor | dav2-indoor-large | K1-scene | 21.25 |
| diode_indoor | dav2-indoor-large | K2 | 279.8 |
| diode_indoor | dav2-indoor-large | K3-depth | 799.1 |
| diode_indoor | dav2-indoor-large | K3-edge | 783.8 |
| diode_indoor | dav2-indoor-large | RCPS | 328.9 |
| diode_indoor | dav2-indoor-large | V4 | 14.46 |
| diode_outdoor | dav2-outdoor-large | K0 | 36.9 |
| diode_outdoor | dav2-outdoor-large | K1 | 922.7 |
| diode_outdoor | dav2-outdoor-large | K1-scene | 24.74 |
| diode_outdoor | dav2-outdoor-large | K2 | 338.5 |
| diode_outdoor | dav2-outdoor-large | K3-depth | 1022 |
| diode_outdoor | dav2-outdoor-large | K3-edge | 1005 |
| diode_outdoor | dav2-outdoor-large | RCPS | 589 |
| diode_outdoor | dav2-outdoor-large | V4 | 16.98 |
| kitti | dav2-outdoor-large | K0 | 80.09 |
| kitti | dav2-outdoor-large | K1 | 1779 |
| kitti | dav2-outdoor-large | K1-scene | 1808 |
| kitti | dav2-outdoor-large | K2 | 513.5 |
| kitti | dav2-outdoor-large | K3-depth | 1925 |
| kitti | dav2-outdoor-large | K3-edge | 1898 |
| kitti | dav2-outdoor-large | RCPS | 1758 |
| kitti | dav2-outdoor-large | V4 | 31.51 |
| kitti | unidepthv2-large | K0 | 80.6 |
| kitti | unidepthv2-large | K1 | 1767 |
| kitti | unidepthv2-large | K1-scene | 1799 |
| kitti | unidepthv2-large | K2 | 514.4 |
| kitti | unidepthv2-large | K3-depth | 1906 |
| kitti | unidepthv2-large | K3-edge | 1878 |
| kitti | unidepthv2-large | RCPS | 1749 |
| kitti | unidepthv2-large | V4 | 31.66 |
| nyu | dav2-indoor-large | K0 | 46.47 |
| nyu | dav2-indoor-large | K1 | 1179 |
| nyu | dav2-indoor-large | K1-scene | 1202 |
| nyu | dav2-indoor-large | K2 | 375.3 |
| nyu | dav2-indoor-large | K3-depth | 1273 |
| nyu | dav2-indoor-large | K3-edge | 1280 |
| nyu | dav2-indoor-large | RCPS | 1180 |
| nyu | dav2-indoor-large | V4 | 20.66 |
| nyu | unidepthv2-large | K0 | 47.3 |
| nyu | unidepthv2-large | K1 | 1230 |
| nyu | unidepthv2-large | K1-scene | 1252 |
| nyu | unidepthv2-large | K2 | 381.4 |
| nyu | unidepthv2-large | K3-depth | 1313 |
| nyu | unidepthv2-large | K3-edge | 1326 |
| nyu | unidepthv2-large | RCPS | 1228 |
| nyu | unidepthv2-large | V4 | 21.35 |

## Scale-model fit + predict on cal/test (s, CPU)

| dataset | model | score | scale_fit_s |
|---|---|---|---|
| diode_indoor | dav2-indoor-large | S0-const | 0.01346 |
| diode_indoor | dav2-indoor-large | S1-rel | 0.01413 |
| diode_indoor | dav2-indoor-large | S2-log | 0.0112 |
| diode_indoor | dav2-indoor-large | S3-flip | 7.337 |
| diode_indoor | dav2-indoor-large | S3e-ens | 7.128 |
| diode_indoor | dav2-indoor-large | S4-grad | 7.149 |
| diode_indoor | dav2-indoor-large | S5-learned | 26.68 |
| diode_indoor | dav2-indoor-large | S6-fact | 30.62 |
| diode_indoor | dav2-indoor-large | S6-fact-flip | 7.754 |
| diode_outdoor | dav2-outdoor-large | S0-const | 0.01377 |
| diode_outdoor | dav2-outdoor-large | S1-rel | 0.01794 |
| diode_outdoor | dav2-outdoor-large | S2-log | 0.01191 |
| diode_outdoor | dav2-outdoor-large | S3-flip | 9.197 |
| diode_outdoor | dav2-outdoor-large | S3e-ens | 8.963 |
| diode_outdoor | dav2-outdoor-large | S4-grad | 8.894 |
| diode_outdoor | dav2-outdoor-large | S5-learned | 32.03 |
| diode_outdoor | dav2-outdoor-large | S6-fact | 36.59 |
| diode_outdoor | dav2-outdoor-large | S6-fact-flip | 9.654 |
| kitti | dav2-outdoor-large | S0-const | 0.02736 |
| kitti | dav2-outdoor-large | S1-rel | 0.03181 |
| kitti | dav2-outdoor-large | S2-log | 0.02587 |
| kitti | dav2-outdoor-large | S3-flip | 13.68 |
| kitti | dav2-outdoor-large | S3e-ens | 13.34 |
| kitti | dav2-outdoor-large | S4-grad | 13.38 |
| kitti | dav2-outdoor-large | S5-learned | 51.18 |
| kitti | dav2-outdoor-large | S6-fact | 52.71 |
| kitti | dav2-outdoor-large | S6-fact-flip | 14.12 |
| kitti | unidepthv2-large | S0-const | 0.02708 |
| kitti | unidepthv2-large | S1-rel | 0.03374 |
| kitti | unidepthv2-large | S2-log | 0.02736 |
| kitti | unidepthv2-large | S3-flip | 13.34 |
| kitti | unidepthv2-large | S4c-conf | 13.28 |
| kitti | unidepthv2-large | S5-learned | 47.83 |
| kitti | unidepthv2-large | S6-fact | 49.66 |
| kitti | unidepthv2-large | S6-fact-conf | 13.72 |
| kitti | unidepthv2-large | S6-fact-flip | 13.92 |
| nyu | dav2-indoor-large | S0-const | 0.01459 |
| nyu | dav2-indoor-large | S1-rel | 0.01913 |
| nyu | dav2-indoor-large | S2-log | 0.01501 |
| nyu | dav2-indoor-large | S3-flip | 10.29 |
| nyu | dav2-indoor-large | S3e-ens | 9.889 |
| nyu | dav2-indoor-large | S4-grad | 9.969 |
| nyu | dav2-indoor-large | S5-learned | 44.22 |
| nyu | dav2-indoor-large | S6-fact | 44.67 |
| nyu | dav2-indoor-large | S6-fact-flip | 10.68 |
| nyu | unidepthv2-large | S0-const | 0.01562 |
| nyu | unidepthv2-large | S1-rel | 0.01972 |
| nyu | unidepthv2-large | S2-log | 0.01444 |
| nyu | unidepthv2-large | S3-flip | 10.39 |
| nyu | unidepthv2-large | S4c-conf | 10.15 |
| nyu | unidepthv2-large | S5-learned | 40.47 |
| nyu | unidepthv2-large | S6-fact | 41.39 |
| nyu | unidepthv2-large | S6-fact-conf | 10.69 |
| nyu | unidepthv2-large | S6-fact-flip | 10.81 |
