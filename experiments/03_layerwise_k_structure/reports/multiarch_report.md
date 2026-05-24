# Multi-Architecture Layerwise Accessibility

Dataset: sklearn digits semantic OOD splits. Architectures: MLP, CNN, patch transformer.
Each layer has its own tail `g_ell` to the classifier output. MLP/CNN layers have different dimensions; transformer CLS layers share dimensions and include `delta_forward_1d`.

## Run Info
```text
architecture              split  n_eval  n_ood  num_classes  val_accuracy
         mlp digits_low_vs_high      90     45            5      0.991525
         cnn digits_low_vs_high      90     45            5      0.991525
 transformer digits_low_vs_high      90     45            5      0.957627
         mlp digits_even_vs_odd      90     45            5      0.974138
         cnn digits_even_vs_odd      90     45            5      1.000000
 transformer digits_even_vs_odd      90     45            5      0.939655
```

## Structured vs Random-Adjusted Summary
```text
architecture              split         subspace  k  layer  n  rho_structured_mean  rho_random_mean  rho_adjusted_mean  saturation_rate
         cnn digits_even_vs_odd        state_pca  1      0 90             0.662568         0.730749      -6.818147e-02         0.188889
         cnn digits_even_vs_odd        state_pca  1      1 90             0.666696         0.690839      -2.414372e-02         0.211111
         cnn digits_even_vs_odd        state_pca  1      2 90             0.659170         0.720586      -6.141647e-02         0.177778
         cnn digits_even_vs_odd        state_pca  2      0 90             0.883510         0.932371      -4.886140e-02         0.400000
         cnn digits_even_vs_odd        state_pca  2      1 90             0.859989         0.927698      -6.770861e-02         0.444444
         cnn digits_even_vs_odd        state_pca  2      2 90             0.898749         0.930504      -3.175522e-02         0.444444
         cnn digits_even_vs_odd        state_pca  4      0 90             1.000000         1.000000      -1.145227e-08         1.000000
         cnn digits_even_vs_odd        state_pca  4      1 90             1.000000         1.000000      -1.426814e-09         1.000000
         cnn digits_even_vs_odd        state_pca  4      2 90             1.000000         0.999990       9.943876e-06         1.000000
         cnn digits_even_vs_odd        state_pca  8      0 90             1.000000         1.000000      -5.822503e-16         1.000000
         cnn digits_even_vs_odd        state_pca  8      1 90             1.000000         1.000000      -2.111891e-15         1.000000
         cnn digits_even_vs_odd        state_pca  8      2 90             1.000000         1.000000      -6.167906e-18         1.000000
         cnn digits_low_vs_high        state_pca  1      0 90             0.567798         0.593842      -2.604357e-02         0.077778
         cnn digits_low_vs_high        state_pca  1      1 90             0.564767         0.605361      -4.059432e-02         0.077778
         cnn digits_low_vs_high        state_pca  1      2 90             0.514009         0.594207      -8.019826e-02         0.077778
         cnn digits_low_vs_high        state_pca  2      0 90             0.840475         0.841604      -1.128809e-03         0.288889
         cnn digits_low_vs_high        state_pca  2      1 90             0.867308         0.856489       1.081853e-02         0.244444
         cnn digits_low_vs_high        state_pca  2      2 90             0.803508         0.856988      -5.348029e-02         0.277778
         cnn digits_low_vs_high        state_pca  4      0 90             1.000000         1.000000       4.274605e-14         1.000000
         cnn digits_low_vs_high        state_pca  4      1 90             1.000000         1.000000       1.786842e-14         1.000000
         cnn digits_low_vs_high        state_pca  4      2 90             1.000000         1.000000       5.161303e-15         1.000000
         cnn digits_low_vs_high        state_pca  8      0 90             1.000000         1.000000      -1.171902e-16         1.000000
         cnn digits_low_vs_high        state_pca  8      1 90             1.000000         1.000000      -7.401487e-16         1.000000
         cnn digits_low_vs_high        state_pca  8      2 90             1.000000         1.000000      -9.868649e-18         1.000000
         mlp digits_even_vs_odd        state_pca  1      0 90             0.614632         0.557818       5.681370e-02         0.044444
         mlp digits_even_vs_odd        state_pca  1      1 90             0.373678         0.526849      -1.531711e-01         0.000000
         mlp digits_even_vs_odd        state_pca  1      2 90             0.399920         0.474210      -7.429070e-02         0.022222
         mlp digits_even_vs_odd        state_pca  2      0 90             0.825988         0.874506      -4.851760e-02         0.233333
         mlp digits_even_vs_odd        state_pca  2      1 90             0.770820         0.763464       7.356209e-03         0.111111
         mlp digits_even_vs_odd        state_pca  2      2 90             0.739931         0.766375      -2.644326e-02         0.088889
         mlp digits_even_vs_odd        state_pca  4      0 90             1.000000         1.000000       1.717145e-15         1.000000
         mlp digits_even_vs_odd        state_pca  4      1 90             1.000000         1.000000       1.970399e-14         1.000000
         mlp digits_even_vs_odd        state_pca  4      2 90             1.000000         1.000000       4.107825e-16         1.000000
         mlp digits_even_vs_odd        state_pca  8      0 90             1.000000         1.000000       1.727014e-16         1.000000
         mlp digits_even_vs_odd        state_pca  8      1 90             1.000000         1.000000      -7.401487e-18         1.000000
         mlp digits_even_vs_odd        state_pca  8      2 90             1.000000         1.000000      -6.167906e-17         1.000000
         mlp digits_low_vs_high        state_pca  1      0 90             0.629939         0.606435       2.350477e-02         0.011111
         mlp digits_low_vs_high        state_pca  1      1 90             0.498348         0.513646      -1.529833e-02         0.011111
         mlp digits_low_vs_high        state_pca  1      2 90             0.485709         0.530174      -4.446453e-02         0.000000
         mlp digits_low_vs_high        state_pca  2      0 90             0.917859         0.864787       5.307211e-02         0.344444
         mlp digits_low_vs_high        state_pca  2      1 90             0.889369         0.752123       1.372459e-01         0.088889
         mlp digits_low_vs_high        state_pca  2      2 90             0.892874         0.792156       1.007182e-01         0.100000
         mlp digits_low_vs_high        state_pca  4      0 90             1.000000         1.000000      -1.197684e-14         1.000000
         mlp digits_low_vs_high        state_pca  4      1 90             1.000000         1.000000       3.132803e-14         1.000000
         mlp digits_low_vs_high        state_pca  4      2 90             1.000000         1.000000       6.250408e-13         1.000000
         mlp digits_low_vs_high        state_pca  8      0 90             1.000000         1.000000      -3.108624e-16         1.000000
         mlp digits_low_vs_high        state_pca  8      1 90             1.000000         1.000000      -1.603655e-17         1.000000
         mlp digits_low_vs_high        state_pca  8      2 90             1.000000         1.000000       2.467162e-18         1.000000
 transformer digits_even_vs_odd delta_forward_1d  1      0 90             0.445289         0.342890       1.023983e-01         0.000000
 transformer digits_even_vs_odd delta_forward_1d  1      1 90             0.298595         0.341451      -4.285619e-02         0.000000
 transformer digits_even_vs_odd delta_forward_1d  1      2 90             0.423055         0.354710       6.834544e-02         0.000000
 transformer digits_even_vs_odd        state_pca  1      0 90             0.357248         0.342890       1.435751e-02         0.000000
 transformer digits_even_vs_odd        state_pca  1      1 90             0.264324         0.341451      -7.712710e-02         0.000000
 transformer digits_even_vs_odd        state_pca  1      2 90             0.316777         0.354710      -3.793260e-02         0.000000
 transformer digits_even_vs_odd        state_pca  1      3 90             0.298385         0.394300      -9.591492e-02         0.000000
 transformer digits_even_vs_odd        state_pca  2      0 90             0.624827         0.585470       3.935700e-02         0.000000
 transformer digits_even_vs_odd        state_pca  2      1 90             0.523585         0.594625      -7.104054e-02         0.000000
 transformer digits_even_vs_odd        state_pca  2      2 90             0.518330         0.604400      -8.607018e-02         0.000000
 transformer digits_even_vs_odd        state_pca  2      3 90             0.508801         0.625058      -1.162573e-01         0.000000
 transformer digits_even_vs_odd        state_pca  4      0 90             1.000000         1.000000       1.241266e-13         1.000000
 transformer digits_even_vs_odd        state_pca  4      1 90             1.000000         1.000000       5.689029e-14         1.000000
 transformer digits_even_vs_odd        state_pca  4      2 90             1.000000         1.000000       2.766283e-12         1.000000
 transformer digits_even_vs_odd        state_pca  4      3 90             1.000000         1.000000       2.379233e-13         1.000000
 transformer digits_even_vs_odd        state_pca  8      0 90             1.000000         1.000000       3.454027e-17         1.000000
 transformer digits_even_vs_odd        state_pca  8      1 90             1.000000         1.000000      -1.023872e-16         1.000000
 transformer digits_even_vs_odd        state_pca  8      2 90             1.000000         1.000000      -1.171902e-16         1.000000
 transformer digits_even_vs_odd        state_pca  8      3 90             1.000000         1.000000       1.480297e-17         1.000000
 transformer digits_low_vs_high delta_forward_1d  1      0 90             0.371105         0.380046      -8.941234e-03         0.000000
 transformer digits_low_vs_high delta_forward_1d  1      1 90             0.349922         0.363711      -1.378854e-02         0.000000
 transformer digits_low_vs_high delta_forward_1d  1      2 90             0.374112         0.356768       1.734411e-02         0.000000
 transformer digits_low_vs_high        state_pca  1      0 90             0.391115         0.380046       1.106895e-02         0.000000
 transformer digits_low_vs_high        state_pca  1      1 90             0.391540         0.363711       2.782865e-02         0.000000
 transformer digits_low_vs_high        state_pca  1      2 90             0.344534         0.356768      -1.223413e-02         0.000000
 transformer digits_low_vs_high        state_pca  1      3 90             0.307424         0.352250      -4.482603e-02         0.000000
 transformer digits_low_vs_high        state_pca  2      0 90             0.587364         0.583698       3.666046e-03         0.000000
 transformer digits_low_vs_high        state_pca  2      1 90             0.574990         0.584445      -9.454770e-03         0.000000
 transformer digits_low_vs_high        state_pca  2      2 90             0.553370         0.605981      -5.261070e-02         0.000000
 transformer digits_low_vs_high        state_pca  2      3 90             0.510113         0.583665      -7.355155e-02         0.000000
 transformer digits_low_vs_high        state_pca  4      0 90             1.000000         1.000000       2.271609e-09         1.000000
 transformer digits_low_vs_high        state_pca  4      1 90             1.000000         1.000000       2.019591e-12         1.000000
 transformer digits_low_vs_high        state_pca  4      2 90             1.000000         1.000000      -5.358676e-14         1.000000
 transformer digits_low_vs_high        state_pca  4      3 90             1.000000         1.000000       1.509533e-13         1.000000
 transformer digits_low_vs_high        state_pca  8      0 90             1.000000         1.000000       1.233581e-18         1.000000
 transformer digits_low_vs_high        state_pca  8      1 90             1.000000         1.000000      -1.702342e-16         1.000000
 transformer digits_low_vs_high        state_pca  8      2 90             1.000000         1.000000      -2.467162e-18         1.000000
 transformer digits_low_vs_high        state_pca  8      3 90             1.000000         1.000000      -3.947460e-17         1.000000
```

## Compressibility k50/k80
```text
architecture              split  layer  threshold  n   k_mean  missing_rate
         cnn digits_even_vs_odd      0        0.5 90 1.522222           0.0
         cnn digits_even_vs_odd      0        0.8 90 1.833333           0.0
         cnn digits_even_vs_odd      1        0.5 90 1.533333           0.0
         cnn digits_even_vs_odd      1        0.8 90 1.922222           0.0
         cnn digits_even_vs_odd      2        0.5 90 1.455556           0.0
         cnn digits_even_vs_odd      2        0.8 90 1.844444           0.0
         cnn digits_low_vs_high      0        0.5 90 1.544444           0.0
         cnn digits_low_vs_high      0        0.8 90 2.200000           0.0
         cnn digits_low_vs_high      1        0.5 90 1.433333           0.0
         cnn digits_low_vs_high      1        0.8 90 2.211111           0.0
         cnn digits_low_vs_high      2        0.5 90 1.800000           0.0
         cnn digits_low_vs_high      2        0.8 90 2.355556           0.0
         mlp digits_even_vs_odd      0        0.5 90 1.566667           0.0
         mlp digits_even_vs_odd      0        0.8 90 2.266667           0.0
         mlp digits_even_vs_odd      1        0.5 90 1.955556           0.0
         mlp digits_even_vs_odd      1        0.8 90 2.700000           0.0
         mlp digits_even_vs_odd      2        0.5 90 2.122222           0.0
         mlp digits_even_vs_odd      2        0.8 90 2.888889           0.0
         mlp digits_low_vs_high      0        0.5 90 1.388889           0.0
         mlp digits_low_vs_high      0        0.8 90 1.722222           0.0
         mlp digits_low_vs_high      1        0.5 90 1.522222           0.0
         mlp digits_low_vs_high      1        0.8 90 2.077778           0.0
         mlp digits_low_vs_high      2        0.5 90 1.600000           0.0
         mlp digits_low_vs_high      2        0.8 90 2.055556           0.0
 transformer digits_even_vs_odd      0        0.5 90 2.144444           0.0
 transformer digits_even_vs_odd      0        0.8 90 3.477778           0.0
 transformer digits_even_vs_odd      1        0.5 90 2.588889           0.0
 transformer digits_even_vs_odd      1        0.8 90 3.811111           0.0
 transformer digits_even_vs_odd      2        0.5 90 2.533333           0.0
 transformer digits_even_vs_odd      2        0.8 90 3.811111           0.0
 transformer digits_even_vs_odd      3        0.5 90 2.700000           0.0
 transformer digits_even_vs_odd      3        0.8 90 3.777778           0.0
 transformer digits_low_vs_high      0        0.5 90 2.288889           0.0
 transformer digits_low_vs_high      0        0.8 90 3.544444           0.0
 transformer digits_low_vs_high      1        0.5 90 2.366667           0.0
 transformer digits_low_vs_high      1        0.8 90 3.488889           0.0
 transformer digits_low_vs_high      2        0.5 90 2.411111           0.0
 transformer digits_low_vs_high      2        0.8 90 3.755556           0.0
 transformer digits_low_vs_high      3        0.5 90 2.611111           0.0
 transformer digits_low_vs_high      3        0.8 90 3.833333           0.0
```

## Prompt-Level Best AUCs
```text
architecture              split         subspace   k              metric  best_auc  layer direction   n  delta_entropy_mean  abs_delta_varentropy_mean
         cnn digits_low_vs_high        state_pca 1.0        entropy_mean  0.951111    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 2.0        entropy_mean  0.951111    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 4.0        entropy_mean  0.951111    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 8.0        entropy_mean  0.951111    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 8.0      trace_fim_mean  0.948642    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 4.0      trace_fim_mean  0.946667    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 2.0      trace_fim_mean  0.938272    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 1.0      trace_fim_mean  0.936296    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 1.0      trace_fim_mean  0.928395    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 4.0      trace_fim_mean  0.919012    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 2.0      trace_fim_mean  0.918025    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 1.0        entropy_mean  0.916543    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 2.0        entropy_mean  0.916543    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 4.0        entropy_mean  0.916543    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 8.0        entropy_mean  0.916543    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 8.0      trace_fim_mean  0.915062    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 4.0 rho_structured_mean  0.773580    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 8.0 rho_structured_mean  0.753333    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 4.0 rho_structured_mean  0.735062    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 4.0   rho_adjusted_mean  0.683951    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 2.0 rho_structured_mean  0.683457    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 2.0   rho_adjusted_mean  0.660247    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 8.0 rho_structured_mean  0.658272    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 2.0 rho_structured_mean  0.647901    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 1.0   rho_adjusted_mean  0.615309    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 1.0   rho_adjusted_mean  0.601975    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 1.0 rho_structured_mean  0.600988    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 1.0 rho_structured_mean  0.598519    NaN       NaN NaN                 NaN                        NaN
         cnn digits_even_vs_odd        state_pca 8.0   rho_adjusted_mean  0.570123    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 2.0   rho_adjusted_mean  0.536790    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 4.0   rho_adjusted_mean  0.530123    NaN       NaN NaN                 NaN                        NaN
         cnn digits_low_vs_high        state_pca 8.0   rho_adjusted_mean  0.519012    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 1.0        entropy_mean  0.941235    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 2.0        entropy_mean  0.941235    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 4.0        entropy_mean  0.941235    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 8.0        entropy_mean  0.941235    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 2.0      trace_fim_mean  0.937778    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 4.0      trace_fim_mean  0.936790    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 8.0      trace_fim_mean  0.936790    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 1.0      trace_fim_mean  0.932346    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 1.0        entropy_mean  0.918025    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 2.0        entropy_mean  0.918025    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 4.0        entropy_mean  0.918025    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 8.0        entropy_mean  0.918025    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 8.0      trace_fim_mean  0.893333    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 1.0      trace_fim_mean  0.889877    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 4.0      trace_fim_mean  0.884444    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 2.0      trace_fim_mean  0.867160    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 2.0 rho_structured_mean  0.761975    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 4.0 rho_structured_mean  0.716543    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 1.0 rho_structured_mean  0.695309    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 2.0 rho_structured_mean  0.678519    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 4.0 rho_structured_mean  0.657778    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 2.0   rho_adjusted_mean  0.646420    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 8.0   rho_adjusted_mean  0.638519    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 1.0 rho_structured_mean  0.622222    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 1.0   rho_adjusted_mean  0.604444    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 1.0   rho_adjusted_mean  0.585185    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 2.0   rho_adjusted_mean  0.552593    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 8.0 rho_structured_mean  0.540000    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 8.0   rho_adjusted_mean  0.518765    NaN       NaN NaN                 NaN                        NaN
         mlp digits_low_vs_high        state_pca 4.0   rho_adjusted_mean  0.513086    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 4.0   rho_adjusted_mean  0.507901    NaN       NaN NaN                 NaN                        NaN
         mlp digits_even_vs_odd        state_pca 8.0 rho_structured_mean  0.502716    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 2.0      trace_fim_mean  0.842469    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 1.0      trace_fim_mean  0.836543    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 4.0      trace_fim_mean  0.833580    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 8.0      trace_fim_mean  0.833086    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high delta_forward_1d 1.0        entropy_mean  0.829136    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 1.0        entropy_mean  0.829136    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 2.0        entropy_mean  0.829136    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 4.0        entropy_mean  0.829136    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 8.0        entropy_mean  0.829136    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high delta_forward_1d 1.0      trace_fim_mean  0.824691    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 8.0      trace_fim_mean  0.758025    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd delta_forward_1d 1.0        entropy_mean  0.756049    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 1.0        entropy_mean  0.756049    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 2.0        entropy_mean  0.756049    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 4.0        entropy_mean  0.756049    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 8.0        entropy_mean  0.756049    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 4.0      trace_fim_mean  0.741235    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd delta_forward_1d 1.0      trace_fim_mean  0.710617    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 2.0      trace_fim_mean  0.708642    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 1.0      trace_fim_mean  0.698765    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 8.0 rho_structured_mean  0.678272    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 2.0 rho_structured_mean  0.678025    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 1.0   rho_adjusted_mean  0.646420    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 2.0   rho_adjusted_mean  0.619753    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 2.0 rho_structured_mean  0.617284    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 2.0   rho_adjusted_mean  0.615309    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd delta_forward_1d 1.0 rho_structured_mean  0.609877    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 8.0   rho_adjusted_mean  0.604691    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 4.0 rho_structured_mean  0.601975    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high delta_forward_1d 1.0 rho_structured_mean  0.587160    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 4.0 rho_structured_mean  0.580494    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 8.0 rho_structured_mean  0.569630    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 1.0 rho_structured_mean  0.561975    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 1.0   rho_adjusted_mean  0.557531    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 4.0   rho_adjusted_mean  0.553086    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high        state_pca 8.0   rho_adjusted_mean  0.550123    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 1.0 rho_structured_mean  0.521481    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd        state_pca 4.0   rho_adjusted_mean  0.518519    NaN       NaN NaN                 NaN                        NaN
 transformer digits_even_vs_odd delta_forward_1d 1.0   rho_adjusted_mean  0.512099    NaN       NaN NaN                 NaN                        NaN
 transformer digits_low_vs_high delta_forward_1d 1.0   rho_adjusted_mean  0.506173    NaN       NaN NaN                 NaN                        NaN
```

## Intervention Summary
```text
architecture              split subspace   k       metric  best_auc  layer   direction    n  delta_entropy_mean  abs_delta_varentropy_mean
         cnn digits_even_vs_odd      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.040465                   0.072185
         cnn digits_even_vs_odd      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.032712                   0.058471
         cnn digits_even_vs_odd      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.014041                   0.024216
         cnn digits_low_vs_high      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.075325                   0.108781
         cnn digits_low_vs_high      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.061238                   0.082827
         cnn digits_low_vs_high      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.025074                   0.033528
         mlp digits_even_vs_odd      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.069358                   0.105268
         mlp digits_even_vs_odd      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.038248                   0.061534
         mlp digits_even_vs_odd      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.037712                   0.058137
         mlp digits_low_vs_high      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.084479                   0.127968
         mlp digits_low_vs_high      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.055810                   0.072985
         mlp digits_low_vs_high      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.039123                   0.052315
 transformer digits_even_vs_odd      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.233789                   0.196812
 transformer digits_even_vs_odd      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.064262                   0.062104
 transformer digits_even_vs_odd      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.058769                   0.061967
 transformer digits_even_vs_odd      NaN NaN intervention       NaN    3.0 state_pca_1 90.0            0.037422                   0.048435
 transformer digits_low_vs_high      NaN NaN intervention       NaN    0.0 state_pca_1 90.0            0.110569                   0.189599
 transformer digits_low_vs_high      NaN NaN intervention       NaN    1.0 state_pca_1 90.0            0.085344                   0.127255
 transformer digits_low_vs_high      NaN NaN intervention       NaN    2.0 state_pca_1 90.0            0.066239                   0.089376
 transformer digits_low_vs_high      NaN NaN intervention       NaN    3.0 state_pca_1 90.0            0.040657                   0.056679
```
