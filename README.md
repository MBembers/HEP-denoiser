# high-energy-denoiser

BDT-based denoiser for HEP data using all variables from `ImpactParameterDataset.var_names`.
Trains on MC (signal) vs experimental sideband (background) and saves the model to `models/`.

## Systematic variable testing
Run per-variable BDTs and save ROC/significance plots + summary CSV:

```bash
python -m src.main --systematic-test --systematic-output outputs/systematic
```

## best treshold 

```bash
python -m src.main -v --plots --plot-dir <output dir>
```


## Xc_M and Xb_M optimized treshold 

```bash
python -m src.main -v --plots --plot-dir <output dir> --threshold 0.99743
```

## custom threshold

```bash
python -m src.main -v --plots --plot-dir <output dir> --threshold 0.95
```


## threshold optimized for each variable (SVT output)

```
Systematic variable testing summary
-----------------------------------
    feature_set  n_features      auc  best_significance  best_threshold
            ALL          24 0.997707         232.716203        0.285688
           Xb_M           1 0.983961         231.023808        0.997433
    Xb_IP_OWNPV           1 0.877562         161.735413        0.263217
Xb_IPCHI2_OWNPV           1 0.876015         162.693730        0.237258
           Xc_M           1 0.856943         157.851675        0.276175
  Xb_DIRA_OWNPV           1 0.774505         131.774521        0.192731
          Xc_PT           1 0.691010         122.855198        0.144320
Xc_FDCHI2_OWNPV           1 0.656995         122.582992        0.126737
pi_IPCHI2_OWNPV           1 0.651163         120.902246        0.126043
            p_P           1 0.644734         119.747656        0.130098
 p_IPCHI2_OWNPV           1 0.636820         119.077542        0.140070
 p_IPCHI2_OWNPV           1 0.636820         119.077542        0.140070
          Xb_PT           1 0.633252         116.729406        0.136947
 k_IPCHI2_OWNPV           1 0.617259         118.482136        0.138975
           Xb_P           1 0.613362         117.248862        0.148859

Best AUC: ALL (AUC=0.998, best S/sqrt(S+B)=232.716)

Lowest-AUC variables (candidates to drop):
feature_set      auc  best_significance
        k_P 0.590339         116.956850
 k_IP_OWNPV 0.587613         119.682345
      pi_PT 0.585265         116.708538
pi_IP_OWNPV 0.583177         119.335273
       pi_P 0.559318         116.932714
Systematic summary saved to /home/michal/AGH/sem5/AI_stat/HEP-denoiser/outputs/systematic
    feature_set  n_features      auc  best_significance  best_threshold
            ALL          24 0.997707         232.716203        0.285688
           Xb_M           1 0.983961         231.023808        0.997433
    Xb_IP_OWNPV           1 0.877562         161.735413        0.263217
Xb_IPCHI2_OWNPV           1 0.876015         162.693730        0.237258
           Xc_M           1 0.856943         157.851675        0.276175
  Xb_DIRA_OWNPV           1 0.774505         131.774521        0.192731
          Xc_PT           1 0.691010         122.855198        0.144320
Xc_FDCHI2_OWNPV           1 0.656995         122.582992        0.126737
pi_IPCHI2_OWNPV           1 0.651163         120.902246        0.126043
            p_P           1 0.644734         119.747656        0.130098
```