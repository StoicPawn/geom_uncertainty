# Application: Local Confidence Control

This application filters accessible decoder interventions for cases where uncertainty/confidence moves while the local answer neighborhood is preserved.

Success requires unchanged top-1, top-10 Jaccard at least 0.8, unchanged target correctness when a target exists, and above-median entropy movement.

## Summary
```text
            model     sign  n  success_rate  answer_preserved_rate  abs_delta_entropy_mean  abs_delta_varentropy_mean  abs_delta_confidence_mean  top10_jaccard_mean  top1_changed_rate
Qwen/Qwen2.5-0.5B decrease 36      0.472222               1.000000                0.043186                   0.059525                   0.013663            0.944444           0.000000
Qwen/Qwen2.5-0.5B increase 36      0.444444               0.944444                0.044297                   0.062743                   0.014307            0.929293           0.055556
  microsoft/phi-2 decrease 36      0.444444               1.000000                0.040944                   0.065227                   0.013145            0.949495           0.000000
  microsoft/phi-2 increase 36      0.638889               1.000000                0.046873                   0.078930                   0.014579            0.979798           0.000000
```

## Qualitative Examples
```text
            model                                                      prompt      target               task    topic  layer subspace_family     sign top1_token_before top1_token_after  entropy_before  entropy_after  delta_entropy  confidence_before  confidence_after  delta_confidence  top10_jaccard      rho
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     23       state_pca increase             gifts            gifts        1.464614       1.539323       0.074709           0.654870          0.631673         -0.023197       1.000000 0.921161
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     23       delta_pca increase             gifts            gifts        1.464614       1.537831       0.073217           0.654870          0.631965         -0.022905       1.000000 0.906370
Qwen/Qwen2.5-0.5B               In a school hallway, a student may notice the                generative_open    scene     23       delta_pca increase         following        following        0.857763       0.930576       0.072814           0.805831          0.785606         -0.020224       1.000000 0.956987
Qwen/Qwen2.5-0.5B               In a school hallway, a student may notice the                generative_open    scene     23       state_pca increase         following        following        0.857763       0.929297       0.071534           0.805831          0.786464         -0.019367       1.000000 0.934411
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     23       delta_pca decrease             gifts            gifts        1.464614       1.394564      -0.070050           0.654870          0.676234          0.021364       1.000000 0.906370
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     23       state_pca decrease             gifts            gifts        1.464614       1.394733      -0.069881           0.654870          0.676020          0.021151       1.000000 0.921161
  microsoft/phi-2                       Question: What animal barks?\nAnswer:         dog     token_level_qa   animal     16       delta_pca increase            Answer           Answer        1.278392       1.347720       0.069327           0.544674          0.525810         -0.018864       1.000000 0.896772
  microsoft/phi-2              A traveler entering a city might first see the                generative_open    scene     31       delta_pca increase              city             city        1.753539       1.820301       0.066762           0.560776          0.536895         -0.023882       1.000000 0.885410
  microsoft/phi-2              A traveler entering a city might first see the                generative_open    scene     31       delta_pca decrease              city             city        1.753539       1.687345      -0.066194           0.560776          0.583787          0.023010       1.000000 0.885410
  microsoft/phi-2              A traveler entering a city might first see the                generative_open    scene     31       state_pca increase              city             city        1.753539       1.819343       0.065804           0.560776          0.537081         -0.023695       1.000000 0.859824
  microsoft/phi-2              A traveler entering a city might first see the                generative_open    scene     31       state_pca decrease              city             city        1.753539       1.688321      -0.065218           0.560776          0.583587          0.022811       1.000000 0.859824
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     12       state_pca increase                 珣                珣        1.811423       1.876450       0.065027           0.484166          0.463623         -0.020543       1.000000 0.893070
Qwen/Qwen2.5-0.5B               In a school hallway, a student may notice the                generative_open    scene     23       delta_pca decrease         following        following        0.857763       0.793437      -0.064325           0.805831          0.823372          0.017541       1.000000 0.956987
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     12       delta_pca increase                 珣                珣        1.811423       1.875641       0.064218           0.484166          0.464095         -0.020071       1.000000 0.913709
Qwen/Qwen2.5-0.5B               In a school hallway, a student may notice the                generative_open    scene     23       state_pca decrease         following        following        0.857763       0.793583      -0.064179           0.805831          0.822914          0.017084       1.000000 0.934411
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     12       delta_pca decrease                 珣                珣        1.811423       1.747805      -0.063618           0.484166          0.503721          0.019554       1.000000 0.913709
  microsoft/phi-2                       Question: What animal barks?\nAnswer:         dog     token_level_qa   animal     16       state_pca increase            Answer           Answer        1.278392       1.340330       0.061938           0.544674          0.529281         -0.015393       1.000000 0.951128
  microsoft/phi-2                          A person who teaches students is a     teacher factual_completion     role     31       state_pca increase           teacher          teacher        0.376786       0.438650       0.061864           0.931761          0.918127         -0.013634       0.818182 0.926642
Qwen/Qwen2.5-0.5B             When people want to celebrate, they often bring                generative_open   common     12       state_pca decrease                 珣                珣        1.811423       1.749982      -0.061441           0.484166          0.503306          0.019139       1.000000 0.893070
Qwen/Qwen2.5-0.5B                           The currency used in Japan is the         yen factual_completion currency     23       delta_pca increase               yen              yen        1.118448       1.179324       0.060876           0.499928          0.489881         -0.010048       1.000000 0.931366
  microsoft/phi-2                          A person who teaches students is a     teacher factual_completion     role     31       delta_pca increase           teacher          teacher        0.376786       0.437484       0.060698           0.931761          0.918306         -0.013454       0.818182 0.925239
Qwen/Qwen2.5-0.5B                           The currency used in Japan is the         yen factual_completion currency     23       state_pca increase               yen              yen        1.118448       1.178646       0.060199           0.499928          0.490026         -0.009902       0.818182 0.895640
Qwen/Qwen2.5-0.5B                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     23       delta_pca increase               The              The        1.963751       2.023661       0.059910           0.505743          0.483277         -0.022466       0.818182 0.780248
  microsoft/phi-2                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     31       delta_pca increase           William          William        0.395382       0.455242       0.059860           0.914545          0.899061         -0.015485       1.000000 0.984605
Qwen/Qwen2.5-0.5B                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     23       delta_pca decrease               The              The        1.963751       1.904913      -0.058839           0.505743          0.527021          0.021278       1.000000 0.780248
Qwen/Qwen2.5-0.5B                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     23       state_pca increase               The              The        1.963751       2.022519       0.058768           0.505743          0.485734         -0.020009       1.000000 0.728924
  microsoft/phi-2                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     31       state_pca increase           William          William        0.395382       0.453888       0.058506           0.914545          0.899324         -0.015222       1.000000 0.981662
  microsoft/phi-2 Question: Which planet is known as the Red Planet?\nAnswer:        Mars     token_level_qa  science     16       delta_pca increase               The              The        1.645160       1.701890       0.056730           0.398561          0.385559         -0.013002       1.000000 0.877298
Qwen/Qwen2.5-0.5B                        Question: Who wrote Hamlet?\nAnswer: Shakespeare     token_level_qa  culture     23       state_pca decrease               The              The        1.963751       1.907664      -0.056087           0.505743          0.524228          0.018485       1.000000 0.728924
  microsoft/phi-2 Question: Which planet is known as the Red Planet?\nAnswer:        Mars     token_level_qa  science     16       state_pca increase               The              The        1.645160       1.700950       0.055790           0.398561          0.385531         -0.013030       1.000000 0.712311
```

## Files
```text
selective_confidence_control_summary.csv
selective_confidence_control_examples.csv
report.md
```
