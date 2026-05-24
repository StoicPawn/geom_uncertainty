# Layerwise k-Constrained Random-Adjusted Rho

Definition used in this run:

```text
rho_{ell,k}(B) = ||P_Im(F_ell^{1/2} J_ell B_{ell,k}) F_ell^{1/2} u_ell||^2 / ||F_ell^{1/2} u_ell||^2
rho_adjusted = rho_structured - mean_r rho_random(layer=ell, k)
```

Model: distilbert-base-uncased. Readout/Jacobian: MLM-head logit-lens top-k output lens with k_output=32.
Structured subspaces: state PCA, forward-update PCA, and per-example forward delta direction `h_{ell+1}-h_ell` with k=1.

## Prompt Counts
```text
       topic  correct  error
animal_class        4     12
     antonym        1     19
     capital       97     11
      common        6     10
   continent        0     24
     culture        9      7
    currency       10      6
    language       30      2
        math        1     15
      plural        6     14
        role       12      8
     science        7      9
```

## Mean Structured vs Random-Adjusted Rho
```text
        subspace  k  rho_structured_mean  rho_random_mean  rho_adjusted_mean  rho_adjusted_z_mean
delta_forward_1d  1             0.143595         0.092953       5.064273e-02             0.459479
       delta_pca  1             0.096522         0.092953       3.569734e-03             0.001048
       delta_pca  2             0.202397         0.179170       2.322684e-02             0.211288
       delta_pca  4             0.320918         0.304676       1.624230e-02             0.086246
       delta_pca  8             0.480540         0.490764      -1.022358e-02            -0.093290
       delta_pca 16             0.740137         0.731238       8.898858e-03             0.122157
       delta_pca 32             1.000000         1.000000       1.814405e-15             0.000366
       state_pca  1             0.121413         0.108508       1.290481e-02             0.160383
       state_pca  2             0.226690         0.202618       2.407160e-02             0.208522
       state_pca  4             0.368999         0.336154       3.284583e-02             0.251755
       state_pca  8             0.548693         0.522492       2.620053e-02             0.269906
       state_pca 16             0.777130         0.752375       2.475472e-02             0.321639
       state_pca 32             1.000000         1.000000       4.035016e-15             0.000184
```

## Layer Trajectory Summary
```text
        subspace  k  layer  rho_structured_mean  rho_random_mean  rho_adjusted_mean  error_minus_correct_adjusted
delta_forward_1d  1      0             0.071536         0.062750       8.785476e-03                 -8.250027e-03
delta_forward_1d  1      1             0.129335         0.082403       4.693189e-02                 -1.655008e-02
delta_forward_1d  1      2             0.081881         0.092559      -1.067717e-02                  1.510088e-02
delta_forward_1d  1      3             0.131225         0.102102       2.912279e-02                  4.530961e-02
delta_forward_1d  1      4             0.115396         0.087195       2.820092e-02                  5.585612e-02
delta_forward_1d  1      5             0.332199         0.130707       2.014925e-01                 -2.749088e-01
       delta_pca  1      0             0.060327         0.062750      -2.423802e-03                 -1.042578e-02
       delta_pca  1      1             0.082688         0.082403       2.847157e-04                  9.416831e-03
       delta_pca  1      2             0.079630         0.092559      -1.292879e-02                 -4.738780e-02
       delta_pca  1      3             0.090701         0.102102      -1.140113e-02                  2.122009e-02
       delta_pca  1      4             0.077326         0.087195      -9.869898e-03                  6.165919e-03
       delta_pca  1      5             0.188464         0.130707       5.775731e-02                 -9.257718e-02
       delta_pca  4      0             0.197291         0.215588      -1.829759e-02                  2.525956e-02
       delta_pca  4      1             0.289525         0.277957       1.156817e-02                 -2.324172e-02
       delta_pca  4      2             0.260594         0.340088      -7.949455e-02                  3.456869e-02
       delta_pca  4      3             0.409642         0.313341       9.630044e-02                 -1.054701e-01
       delta_pca  4      4             0.303437         0.286398       1.703891e-02                 -2.557552e-02
       delta_pca  4      5             0.465021         0.394682       7.033843e-02                 -8.572072e-02
       delta_pca 16      0             0.653484         0.682360      -2.887697e-02                 -3.351151e-03
       delta_pca 16      1             0.737091         0.694980       4.211108e-02                 -2.093452e-02
       delta_pca 16      2             0.727628         0.740788      -1.316024e-02                 -1.231750e-02
       delta_pca 16      3             0.784253         0.742107       4.214581e-02                 -5.501289e-02
       delta_pca 16      4             0.707464         0.711650      -4.186264e-03                 -1.571914e-03
       delta_pca 16      5             0.830901         0.815541       1.535973e-02                  3.111261e-05
       delta_pca 32      0             1.000000         1.000000      -1.991463e-15                 -2.262426e-15
       delta_pca 32      1             1.000000         1.000000       5.950448e-15                  1.314215e-14
       delta_pca 32      2             1.000000         1.000000       3.041317e-15                  6.973437e-15
       delta_pca 32      3             1.000000         1.000000       8.430756e-17                 -2.170986e-15
       delta_pca 32      4             1.000000         1.000000       1.225409e-15                  3.494278e-15
       delta_pca 32      5             1.000000         1.000000       2.576411e-15                 -1.488277e-15
       state_pca  1      0             0.050504         0.062750      -1.224598e-02                  2.869024e-02
       state_pca  1      1             0.096538         0.082403       1.413522e-02                 -1.536373e-02
       state_pca  1      2             0.126808         0.092559       3.424929e-02                 -3.141724e-02
       state_pca  1      3             0.133787         0.102102       3.168548e-02                 -1.407249e-03
       state_pca  1      4             0.109655         0.087195       2.245956e-02                 -2.217184e-02
       state_pca  1      5             0.184504         0.130707       5.379742e-02                 -1.305819e-01
       state_pca  1      6             0.148092         0.201839      -5.374729e-02                  8.352159e-02
       state_pca  4      0             0.214116         0.215588      -1.472351e-03                  5.065087e-02
       state_pca  4      1             0.328539         0.277957       5.058162e-02                 -9.320015e-02
       state_pca  4      2             0.360316         0.340088       2.022767e-02                  6.854723e-02
       state_pca  4      3             0.359873         0.313341       4.653216e-02                  1.844948e-03
       state_pca  4      4             0.363187         0.286398       7.678902e-02                 -3.922052e-02
       state_pca  4      5             0.461037         0.394682       6.635519e-02                 -9.832074e-02
       state_pca  4      6             0.495927         0.525020      -2.909248e-02                  2.245998e-02
       state_pca 16      0             0.672843         0.682360      -9.517346e-03                  2.306676e-02
       state_pca 16      1             0.734503         0.694980       3.952225e-02                 -4.026049e-02
       state_pca 16      2             0.789498         0.740788       4.870954e-02                 -9.867261e-03
       state_pca 16      3             0.766898         0.742107       2.479048e-02                 -2.224884e-02
       state_pca 16      4             0.743061         0.711650       3.141063e-02                  1.309019e-02
       state_pca 16      5             0.846292         0.815541       3.075038e-02                 -1.609262e-03
       state_pca 16      6             0.886814         0.879197       7.617099e-03                  2.409133e-02
       state_pca 32      0             1.000000         1.000000       1.963707e-16                  1.157473e-16
       state_pca 32      1             1.000000         1.000000       4.646283e-15                  1.175813e-14
       state_pca 32      2             1.000000         1.000000       1.003364e-15                  1.999278e-15
       state_pca 32      3             1.000000         1.000000       6.709910e-16                 -1.601270e-15
       state_pca 32      4             1.000000         1.000000       1.731254e-16                  4.264455e-15
       state_pca 32      5             1.000000         1.000000       1.994932e-15                 -7.180514e-16
       state_pca 32      6             1.000000         1.000000       1.956005e-14                 -2.558338e-14
```

## Predictive Feature AUCs
```text
 split_kind                      feature_set  n_splits  auc_mean  auc_std  auc_min  auc_max
 fact_group     scalar_plus_adj_state_pca_k2        80  0.895513 0.033054 0.793258 0.950893
 fact_group scalar_plus_adj_delta_forward_1d        80  0.891376 0.031379 0.792570 0.949223
 fact_group     scalar_plus_adj_state_pca_k4        80  0.889954 0.032270 0.791538 0.947133
 fact_group             scalar_layer_profile        80  0.886463 0.034536 0.769522 0.938776
 fact_group     scalar_plus_adj_delta_pca_k8        80  0.885798 0.032347 0.770554 0.947133
 fact_group     scalar_plus_adj_delta_pca_k1        80  0.884556 0.034416 0.779842 0.938769
 fact_group    scalar_plus_adj_delta_pca_k16        80  0.884171 0.035230 0.755418 0.950492
 fact_group     scalar_plus_adj_state_pca_k1        80  0.883660 0.034298 0.775026 0.941311
 fact_group    scalar_plus_adj_delta_pca_k32        80  0.883200 0.033429 0.765738 0.931895
 fact_group     scalar_plus_adj_delta_pca_k4        80  0.882771 0.032547 0.780186 0.939068
 fact_group     scalar_plus_adj_delta_pca_k2        80  0.881945 0.032943 0.778466 0.942623
 fact_group     scalar_plus_adj_state_pca_k8        80  0.881507 0.037267 0.768146 0.945938
 fact_group    scalar_plus_adj_state_pca_k16        80  0.877343 0.034448 0.761266 0.947213
 fact_group    scalar_plus_adj_state_pca_k32        80  0.870722 0.035408 0.765738 0.940051
 fact_group                rho_state_pca_k16        80  0.857549 0.030493 0.795281 0.933393
 fact_group                 rho_state_pca_k8        80  0.848244 0.032652 0.774163 0.917989
 fact_group                 rho_state_pca_k4        80  0.842868 0.034496 0.747768 0.916069
 fact_group                 rho_state_pca_k2        80  0.838280 0.038788 0.751199 0.919016
 fact_group                 rho_state_pca_k1        80  0.778248 0.047785 0.670779 0.877176
 fact_group         rho_adj_delta_forward_1d        80  0.775901 0.040323 0.675649 0.897551
 fact_group                 rho_delta_pca_k8        80  0.775430 0.038818 0.680656 0.862306
 fact_group                 rho_delta_pca_k4        80  0.769212 0.044651 0.668917 0.888074
 fact_group                rho_delta_pca_k16        80  0.760198 0.043757 0.666779 0.853741
 fact_group             rho_delta_forward_1d        80  0.758766 0.042416 0.645455 0.904719
 fact_group                 rho_delta_pca_k2        80  0.749908 0.044781 0.636721 0.850294
 fact_group             rho_adj_delta_pca_k2        80  0.745752 0.048468 0.622642 0.839218
 fact_group             rho_adj_delta_pca_k4        80  0.737772 0.048996 0.577628 0.824390
 fact_group             rho_adj_delta_pca_k8        80  0.733790 0.041594 0.611148 0.838777
 fact_group             rho_adj_state_pca_k1        80  0.731556 0.051197 0.620177 0.830508
 fact_group             rho_adj_state_pca_k4        80  0.720746 0.044336 0.615117 0.846875
 fact_group             rho_adj_state_pca_k2        80  0.679236 0.049813 0.571474 0.797436
 fact_group                 rho_delta_pca_k1        80  0.678842 0.048792 0.557943 0.822879
 fact_group            rho_adj_delta_pca_k16        80  0.672345 0.046739 0.583004 0.801020
 fact_group            rho_adj_state_pca_k16        80  0.664897 0.045394 0.582633 0.797192
 fact_group             rho_adj_delta_pca_k1        80  0.653394 0.056492 0.530677 0.790625
 fact_group             rho_adj_state_pca_k8        80  0.629857 0.043604 0.493099 0.745625
 fact_group            rho_adj_delta_pca_k32        80  0.580945 0.047487 0.477058 0.676367
 fact_group            rho_adj_state_pca_k32        80  0.521384 0.045984 0.397980 0.642764
 fact_group                rho_state_pca_k32        80  0.507479 0.033285 0.406349 0.601010
 fact_group                rho_delta_pca_k32        80  0.500000 0.000000 0.500000 0.500000
leave_topic                 rho_state_pca_k2        11  0.647800 0.290514 0.000000 0.916667
leave_topic                rho_state_pca_k16        11  0.629817 0.229626 0.157895 0.950000
leave_topic     scalar_plus_adj_state_pca_k2        11  0.620259 0.234712 0.157895 0.959700
leave_topic     scalar_plus_adj_state_pca_k8        11  0.607378 0.192692 0.333333 0.948454
leave_topic     scalar_plus_adj_state_pca_k1        11  0.604200 0.207329 0.266667 0.938144
leave_topic             scalar_layer_profile        11  0.604199 0.205732 0.315789 0.965323
leave_topic    scalar_plus_adj_delta_pca_k16        11  0.601282 0.211062 0.266667 0.956888
leave_topic     scalar_plus_adj_state_pca_k4        11  0.600393 0.209755 0.263158 0.922212
leave_topic    scalar_plus_adj_state_pca_k16        11  0.590062 0.204484 0.333333 0.931584
leave_topic                 rho_state_pca_k8        11  0.585084 0.255965 0.000000 0.866667
leave_topic     scalar_plus_adj_delta_pca_k1        11  0.584406 0.209958 0.266667 0.968135
leave_topic     scalar_plus_adj_delta_pca_k2        11  0.583756 0.197744 0.315789 0.951265
leave_topic    scalar_plus_adj_delta_pca_k32        11  0.578437 0.256346 0.000000 0.948454
leave_topic                 rho_delta_pca_k2        11  0.577691 0.183598 0.312500 0.850000
leave_topic     scalar_plus_adj_delta_pca_k8        11  0.575976 0.214485 0.133333 0.964386
leave_topic    scalar_plus_adj_state_pca_k32        11  0.573664 0.260614 0.000000 0.943768
leave_topic             rho_adj_state_pca_k2        11  0.566395 0.175447 0.260417 0.866667
leave_topic                 rho_state_pca_k4        11  0.560547 0.292821 0.000000 0.966667
leave_topic scalar_plus_adj_delta_forward_1d        11  0.560424 0.287102 0.066667 0.978444
leave_topic     scalar_plus_adj_delta_pca_k4        11  0.557167 0.223528 0.200000 0.945642
leave_topic            rho_adj_delta_pca_k16        11  0.543601 0.142880 0.270833 0.714286
leave_topic            rho_adj_state_pca_k16        11  0.541946 0.253119 0.233333 1.000000
leave_topic             rho_adj_state_pca_k4        11  0.541232 0.211230 0.214286 0.916667
leave_topic             rho_adj_state_pca_k1        11  0.538266 0.232332 0.150000 1.000000
leave_topic             rho_adj_delta_pca_k2        11  0.534906 0.226620 0.183333 0.894737
leave_topic                 rho_state_pca_k1        11  0.529946 0.170966 0.157895 0.739583
leave_topic                 rho_delta_pca_k8        11  0.510114 0.201820 0.000000 0.766667
leave_topic                 rho_delta_pca_k1        11  0.503742 0.210007 0.083333 0.761905
leave_topic                rho_delta_pca_k32        11  0.500000 0.000000 0.500000 0.500000
leave_topic                rho_delta_pca_k16        11  0.498913 0.205490 0.000000 0.743205
leave_topic                rho_state_pca_k32        11  0.495229 0.015088 0.447516 0.500000
leave_topic             rho_adj_delta_pca_k4        11  0.487855 0.230994 0.000000 0.857143
leave_topic            rho_adj_delta_pca_k32        11  0.484366 0.226496 0.000000 0.816667
leave_topic                 rho_delta_pca_k4        11  0.472213 0.200836 0.133333 0.762887
leave_topic             rho_adj_state_pca_k8        11  0.460364 0.222257 0.142857 0.800000
leave_topic             rho_adj_delta_pca_k1        11  0.459226 0.226019 0.062500 0.800000
leave_topic             rho_adj_delta_pca_k8        11  0.458106 0.226914 0.000000 0.797619
leave_topic             rho_delta_forward_1d        11  0.455659 0.245428 0.000000 0.842549
leave_topic         rho_adj_delta_forward_1d        11  0.397932 0.232928 0.000000 0.669166
leave_topic            rho_adj_state_pca_k32        11  0.365200 0.230184 0.000000 0.729167
```

## Paired AUC Deltas
```text
 split_kind                             left                right  left_auc_mean  right_auc_mean  delta_auc_mean
 fact_group         rho_adj_delta_forward_1d rho_delta_forward_1d       0.775901        0.758766    1.713444e-02
 fact_group         rho_adj_delta_forward_1d scalar_layer_profile       0.775901        0.886463   -1.105619e-01
 fact_group scalar_plus_adj_delta_forward_1d scalar_layer_profile       0.891376        0.886463    4.913720e-03
 fact_group             rho_adj_state_pca_k1     rho_state_pca_k1       0.731556        0.778248   -4.669167e-02
 fact_group             rho_adj_state_pca_k1 scalar_layer_profile       0.731556        0.886463   -1.549061e-01
 fact_group     scalar_plus_adj_state_pca_k1 scalar_layer_profile       0.883660        0.886463   -2.803061e-03
 fact_group             rho_adj_state_pca_k2     rho_state_pca_k2       0.679236        0.838280   -1.590441e-01
 fact_group             rho_adj_state_pca_k2 scalar_layer_profile       0.679236        0.886463   -2.072268e-01
 fact_group     scalar_plus_adj_state_pca_k2 scalar_layer_profile       0.895513        0.886463    9.050784e-03
 fact_group             rho_adj_state_pca_k4     rho_state_pca_k4       0.720746        0.842868   -1.221224e-01
 fact_group             rho_adj_state_pca_k4 scalar_layer_profile       0.720746        0.886463   -1.657170e-01
 fact_group     scalar_plus_adj_state_pca_k4 scalar_layer_profile       0.889954        0.886463    3.491847e-03
 fact_group             rho_adj_state_pca_k8     rho_state_pca_k8       0.629857        0.848244   -2.183868e-01
 fact_group             rho_adj_state_pca_k8 scalar_layer_profile       0.629857        0.886463   -2.566057e-01
 fact_group     scalar_plus_adj_state_pca_k8 scalar_layer_profile       0.881507        0.886463   -4.956012e-03
 fact_group            rho_adj_state_pca_k16    rho_state_pca_k16       0.664897        0.857549   -1.926513e-01
 fact_group            rho_adj_state_pca_k16 scalar_layer_profile       0.664897        0.886463   -2.215652e-01
 fact_group    scalar_plus_adj_state_pca_k16 scalar_layer_profile       0.877343        0.886463   -9.119296e-03
 fact_group            rho_adj_state_pca_k32    rho_state_pca_k32       0.521384        0.507479    1.390545e-02
 fact_group            rho_adj_state_pca_k32 scalar_layer_profile       0.521384        0.886463   -3.650782e-01
 fact_group    scalar_plus_adj_state_pca_k32 scalar_layer_profile       0.870722        0.886463   -1.574074e-02
 fact_group             rho_adj_delta_pca_k1     rho_delta_pca_k1       0.653394        0.678842   -2.544814e-02
 fact_group             rho_adj_delta_pca_k1 scalar_layer_profile       0.653394        0.886463   -2.330688e-01
 fact_group     scalar_plus_adj_delta_pca_k1 scalar_layer_profile       0.884556        0.886463   -1.906397e-03
 fact_group             rho_adj_delta_pca_k2     rho_delta_pca_k2       0.745752        0.749908   -4.156252e-03
 fact_group             rho_adj_delta_pca_k2 scalar_layer_profile       0.745752        0.886463   -1.407108e-01
 fact_group     scalar_plus_adj_delta_pca_k2 scalar_layer_profile       0.881945        0.886463   -4.517127e-03
 fact_group             rho_adj_delta_pca_k4     rho_delta_pca_k4       0.737772        0.769212   -3.143944e-02
 fact_group             rho_adj_delta_pca_k4 scalar_layer_profile       0.737772        0.886463   -1.486903e-01
 fact_group     scalar_plus_adj_delta_pca_k4 scalar_layer_profile       0.882771        0.886463   -3.691528e-03
 fact_group             rho_adj_delta_pca_k8     rho_delta_pca_k8       0.733790        0.775430   -4.164040e-02
 fact_group             rho_adj_delta_pca_k8 scalar_layer_profile       0.733790        0.886463   -1.526726e-01
 fact_group     scalar_plus_adj_delta_pca_k8 scalar_layer_profile       0.885798        0.886463   -6.648868e-04
 fact_group            rho_adj_delta_pca_k16    rho_delta_pca_k16       0.672345        0.760198   -8.785316e-02
 fact_group            rho_adj_delta_pca_k16 scalar_layer_profile       0.672345        0.886463   -2.141181e-01
 fact_group    scalar_plus_adj_delta_pca_k16 scalar_layer_profile       0.884171        0.886463   -2.291984e-03
 fact_group            rho_adj_delta_pca_k32    rho_delta_pca_k32       0.580945        0.500000    8.094531e-02
 fact_group            rho_adj_delta_pca_k32 scalar_layer_profile       0.580945        0.886463   -3.055173e-01
 fact_group    scalar_plus_adj_delta_pca_k32 scalar_layer_profile       0.883200        0.886463   -3.262697e-03
leave_topic         rho_adj_delta_forward_1d rho_delta_forward_1d       0.397932        0.455659   -5.772641e-02
leave_topic         rho_adj_delta_forward_1d scalar_layer_profile       0.397932        0.604199   -2.062671e-01
leave_topic scalar_plus_adj_delta_forward_1d scalar_layer_profile       0.560424        0.604199   -4.377567e-02
leave_topic             rho_adj_state_pca_k1     rho_state_pca_k1       0.538266        0.529946    8.319313e-03
leave_topic             rho_adj_state_pca_k1 scalar_layer_profile       0.538266        0.604199   -6.593351e-02
leave_topic     scalar_plus_adj_state_pca_k1 scalar_layer_profile       0.604200        0.604199    3.211929e-07
leave_topic             rho_adj_state_pca_k2     rho_state_pca_k2       0.566395        0.647800   -8.140553e-02
leave_topic             rho_adj_state_pca_k2 scalar_layer_profile       0.566395        0.604199   -3.780474e-02
leave_topic     scalar_plus_adj_state_pca_k2 scalar_layer_profile       0.620259        0.604199    1.606005e-02
leave_topic             rho_adj_state_pca_k4     rho_state_pca_k4       0.541232        0.560547   -1.931512e-02
leave_topic             rho_adj_state_pca_k4 scalar_layer_profile       0.541232        0.604199   -6.296764e-02
leave_topic     scalar_plus_adj_state_pca_k4 scalar_layer_profile       0.600393        0.604199   -3.806733e-03
leave_topic             rho_adj_state_pca_k8     rho_state_pca_k8       0.460364        0.585084   -1.247200e-01
leave_topic             rho_adj_state_pca_k8 scalar_layer_profile       0.460364        0.604199   -1.438354e-01
leave_topic     scalar_plus_adj_state_pca_k8 scalar_layer_profile       0.607378        0.604199    3.178927e-03
leave_topic            rho_adj_state_pca_k16    rho_state_pca_k16       0.541946        0.629817   -8.787141e-02
leave_topic            rho_adj_state_pca_k16 scalar_layer_profile       0.541946        0.604199   -6.225368e-02
leave_topic    scalar_plus_adj_state_pca_k16 scalar_layer_profile       0.590062        0.604199   -1.413751e-02
leave_topic            rho_adj_state_pca_k32    rho_state_pca_k32       0.365200        0.495229   -1.300292e-01
leave_topic            rho_adj_state_pca_k32 scalar_layer_profile       0.365200        0.604199   -2.389997e-01
leave_topic    scalar_plus_adj_state_pca_k32 scalar_layer_profile       0.573664        0.604199   -3.053532e-02
leave_topic             rho_adj_delta_pca_k1     rho_delta_pca_k1       0.459226        0.503742   -4.451577e-02
leave_topic             rho_adj_delta_pca_k1 scalar_layer_profile       0.459226        0.604199   -1.449732e-01
leave_topic     scalar_plus_adj_delta_pca_k1 scalar_layer_profile       0.584406        0.604199   -1.979310e-02
leave_topic             rho_adj_delta_pca_k2     rho_delta_pca_k2       0.534906        0.577691   -4.278471e-02
leave_topic             rho_adj_delta_pca_k2 scalar_layer_profile       0.534906        0.604199   -6.929340e-02
leave_topic     scalar_plus_adj_delta_pca_k2 scalar_layer_profile       0.583756        0.604199   -2.044287e-02
leave_topic             rho_adj_delta_pca_k4     rho_delta_pca_k4       0.487855        0.472213    1.564146e-02
leave_topic             rho_adj_delta_pca_k4 scalar_layer_profile       0.487855        0.604199   -1.163446e-01
leave_topic     scalar_plus_adj_delta_pca_k4 scalar_layer_profile       0.557167        0.604199   -4.703206e-02
leave_topic             rho_adj_delta_pca_k8     rho_delta_pca_k8       0.458106        0.510114   -5.200828e-02
leave_topic             rho_adj_delta_pca_k8 scalar_layer_profile       0.458106        0.604199   -1.460932e-01
leave_topic     scalar_plus_adj_delta_pca_k8 scalar_layer_profile       0.575976        0.604199   -2.822373e-02
leave_topic            rho_adj_delta_pca_k16    rho_delta_pca_k16       0.543601        0.498913    4.468783e-02
leave_topic            rho_adj_delta_pca_k16 scalar_layer_profile       0.543601        0.604199   -6.059814e-02
leave_topic    scalar_plus_adj_delta_pca_k16 scalar_layer_profile       0.601282        0.604199   -2.917543e-03
leave_topic            rho_adj_delta_pca_k32    rho_delta_pca_k32       0.484366        0.500000   -1.563401e-02
leave_topic            rho_adj_delta_pca_k32 scalar_layer_profile       0.484366        0.604199   -1.198333e-01
leave_topic    scalar_plus_adj_delta_pca_k32 scalar_layer_profile       0.578437        0.604199   -2.576227e-02
```
