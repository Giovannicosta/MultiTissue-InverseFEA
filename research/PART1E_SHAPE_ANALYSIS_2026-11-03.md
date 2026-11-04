# Part1_E shape analysis and geometry denoising (exploratory report)

**Recorded:** 2026-11-03  
**Working notebook:** `04_Shape_Analysis_Comparison.ipynb` (Google Colab; **not yet committed to this repository**)  
**Research objective:** Predict `Part1_E` from geometry while improving robustness to noisy observed contours/coordinates.  
**Status:** Steps 1–31 reported and discussed; Step 32 (out-of-fold reconstruction-aware predictor) proposed but **no results reported**.

## Executive summary

The original clean-geometry problem is largely solved by the tested Random Forest models (R² about 0.98–0.997). The main obstacle is **noise robustness**. Clean-trained models collapse under radial contour noise with bottom-coordinate jitter. Noise-trained specialists help, and combining **geometry denoising with a clean-trained predictor** produced the strongest result **in notebook 04 on the explored validation split**:

| Method, all evaluated on the same noisy validation rows | Part1_E R² | Part1_E MAE |
| --- | ---: | ---: |
| Clean-trained Bottom + Inner RF, directly on noise | -1.254235 | 0.021212 |
| Noise-trained Bottom + Inner RF specialist | 0.507630 | 0.010204 |
| **ExtraTrees geometry denoiser -> clean-trained Bottom + Inner RF** | **0.632186** | **0.008694** |

Relative to the radial specialist, denoising increased R² by 0.124556 and lowered prediction MAE by approximately **14.8%**. Mean absolute *geometry coordinate* error fell from 0.021900 to 0.001923 (about **91.2%** lower). Despite this, predictions still regress toward the middle of the `Part1_E` range, particularly at 0.10–0.11 and 0.15–0.16.

**Interpretation:** Reconstructing clean geometry can restore useful prediction information that is lost when a clean-trained predictor receives noisy coordinates. Lower coordinate MAE does **not** guarantee proportional improvement in modulus prediction.

## 1. Data, features, pipeline, and evaluation scope

- Dataset: 6,510 existing simulation rows, 99 columns, including target `Part1_E`. No new FEA simulations were generated.
- Geometry: 54 columns in three 18-coordinate blocks: **Bottom** (`inner_y1..9`, `inner_z1..9`), **Inner** (`innerShape_x1..9`, `innerShape_y1..9`), and **Outer** (`outerShape_x1..9`, `outerShape_y1..9`).
- Preprocessing: `geometry_preprocessing.preprocess_geometry` performs periodic cubic-spline processing on the nine-point Inner/Outer contours; Bottom coordinates are retained. It uses uniform sample-index parameterization, not chord length.
- Experiment split: initial 80/20 split with `random_state=42` gives training pool **5,208** and initially reserved test **1,302**. Later, the training pool was split 75/25 (seed 42) into **3,906 training** and **1,302 validation** rows. The original 1,302-row test was already inspected in notebook Steps 3–7, so it is **not a genuinely pristine final holdout**.
- Most later comparisons use the 3,906 / 1,302 train/validation split. Early clean PCA results from Steps 3–7 use the earlier train/test comparison and must **not** be treated as directly equivalent.
- Unless otherwise noted, regressors use `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
- Radial corruption: `add_geometry_noise(method="radial", noise_level=0.05, bottom_jitter=0.01, harmonics=3)`; one validation realization used seed 202. Training specialists used four independent radial-noise realizations of each training simulation (augmented rows, **not** independent simulations).
- Noise intensity is tied to implemented coordinate perturbations. In particular, `noise_level=0` still produces Bottom jitter unless `bottom_jitter=0`. Gaussian and radial levels should be compared using realized displacement magnitudes, not only parameter values.

## 2. Clean prediction and PCA experiments (Steps 3–8)

Initial clean-geometry comparison (early 80/20 split):

| Representation | Features | R² | MAE |
| --- | ---: | ---: | ---: |
| Raw geometry | 54 | 0.987102 | 0.001451 |
| Separate PCA, 6 PCs per group | 18 | **0.996604** | **0.000692** |
| Joint PCA | 18 | 0.991567 | 0.000961 |

Clean-data component budget for separate PCA: 7 PCs (3 Bottom, 2 Inner, 2 Outer) R² 0.972780 / MAE 0.002138; 10 PCs (4,3,3) R² 0.995793 / MAE 0.000751; 18 PCs (6,6,6) R² 0.996604 / MAE 0.000692. First PCA component explained roughly 73.4% of Bottom, 87.7% of Inner, and 93.0% of Outer standardized within-group variation. Four joint PCs accounted for about 99.86% of standardized joint variance; high geometric variance retention did not by itself ensure robustness.

Clean-trained models on the later validation split:

| Input representation | Clean R² | Radial R² | Clean MAE | Radial MAE |
| --- | ---: | ---: | ---: | ---: |
| Raw 54 | 0.984254 | -1.232819 | 0.001611 | 0.021089 |
| Separate PCA 10 | 0.995427 | -1.545616 | 0.000802 | 0.022694 |
| Separate PCA 18 | 0.996439 | -1.490346 | 0.000741 | 0.022420 |

**Finding:** PCA is excellent for clean-data compression, but clean-trained PCA pipelines were **not** robust to this noisy validation condition.

## 3. Noise specialists and region sensitivity (Steps 11–17)

Four-realization augmentation was used to train Gaussian, radial, and mixed specialists. Selected results:

| Specialist | Evaluation condition | R² | MAE |
| --- | --- | ---: | ---: |
| Gaussian | Gaussian | 0.053183 | 0.014505 |
| Radial | Clean | 0.568958 | 0.009198 |
| Radial | Radial | 0.474209 | 0.010553 |
| Mixed | Radial | 0.191078 | 0.013363 |

These are specific training/noise configurations, not an exhaustive test of each noise family. Radial-only training worked best among those listed for radial validation.

**Single-group corruption of the clean-trained 54-feature model** (other groups clean):

| Condition | R² | MAE |
| --- | ---: | ---: |
| Clean | 0.984254 | 0.001611 |
| Bottom only noisy | -0.440418 | 0.016175 |
| Inner only noisy | **-1.456713** | 0.022365 |
| Outer only noisy | 0.945794 | 0.003023 |
| All groups noisy | -1.232819 | 0.021089 |

Inner-only noise caused the largest deterioration in this model. **This does not mean that Inner is intrinsically most sensitive in every model**; feature scales and fitted relationships matter.

An **Inner-only-noise specialist**, trained on corrupted Inner with clean Bottom/Outer, achieved R² 0.968156 / MAE 0.002167 under Inner-only noise, but R² -1.461343 / MAE 0.022274 when all groups were noisy. This demonstrates **dependence on the other clean groups**, not that Inner corruption was solved in general.

For the Inner-only-noise specialist, two-component linear PCA, six-component PCA, RBF Kernel PCA (2/6), and a two-dimensional dense autoencoder gave similar results on the matched Inner-only-noise validation (about R² 0.9687–0.9693). None solved all-group noise. A small number of tested hyperparameter settings is not evidence that nonlinear methods are universally inferior.

PCA score stability under radial noise was uneven: Inner PC1 clean–noisy correlation **0.976774**, Inner PC2 **0.357149**; Outer PC1 **0.605839**; Bottom PCs were especially unstable under the tested jitter and standardization. Important variation for reconstruction or material prediction is not necessarily aligned with the largest PCA components.

## 4. Interpretable descriptors (Steps 18–22)

Contour inspection indicated Inner PC1 roughly tracks contour expansion/contraction, while PC2 captures smaller deformations. Five Inner contour descriptors were calculated on nine-point polygons:

| Descriptor | Clean–noisy correlation | Normalized MAE |
| --- | ---: | ---: |
| Area | 0.999996 | 0.009715 |
| Perimeter | 0.999918 | 0.025956 |
| Circularity | 0.072995 | 7.714007 |
| Mean radius | **0.999999** | **0.003268** |
| Radius standard deviation | 0.291299 | 4.388892 |

Area, perimeter and mean radius are very stable under the tested corruption, but a Random Forest using **only those three Inner descriptors** failed to predict the modulus: clean R² **-0.347279** / MAE **0.016658**, radial R² **-0.107530** / MAE **0.015529**. The Inner-only and all-radial-noise results are identical because both inputs contain the same noisy Inner coordinates and the predictor receives no Bottom/Outer coordinates.

**Finding:** Noise-invariant shape summaries may discard crucial material-property information. The very high normalized errors for circularity/radius SD may also reflect extremely small clean standard deviations, not solely large absolute perturbations.

## 5. Geometry group ablation and Bottom + Inner models (Steps 23–26)

Clean validation group ablation with the same Random Forest setup:

| Input groups | Coordinates | R² | MAE |
| --- | ---: | ---: | ---: |
| Bottom only | 18 | 0.810109 | 0.005621 |
| Inner only | 18 | 0.890747 | 0.004517 |
| Outer only | 18 | 0.728316 | 0.007159 |
| **Bottom + Inner** | **36** | **0.984360** | **0.001608** |
| Bottom + Outer | 36 | 0.965365 | 0.002198 |
| Inner + Outer | 36 | 0.905436 | 0.004176 |
| All geometry | 54 | 0.984254 | 0.001611 |

Inner is strongest individually; Bottom + Inner was the best tested clean combination and matched the 54-feature model closely. Removing Outer also helped the radial specialist:

| Radial-noise-trained model | Clean R² | Radial R² | Radial MAE |
| --- | ---: | ---: | ---: |
| Bottom + Inner, 36 coordinates | **0.586686** | **0.507630** | **0.010204** |
| All geometry, 54 coordinates | 0.568958 | 0.474209 | 0.010553 |

Here, the 36-feature radial specialist reduced radial MAE by about **3.3%** versus the 54-feature specialist. This is exploratory, not a statistical significance test.

Joint Bottom + Inner PCA did **not** improve radial performance:

| Representation | Radial R² | Radial MAE |
| --- | ---: | ---: |
| Raw Bottom + Inner, 36 | **0.507630** | **0.010204** |
| Joint PCA 4, clean fitted | -0.037226 | 0.014979 |
| Joint PCA 8, clean fitted | 0.004371 | 0.014834 |
| Joint PCA 16, clean fitted | 0.022635 | 0.014738 |
| Joint PCA 16, noise fitted | 0.094144 | 0.014120 |
| Joint PCA 16, mixed fitted | 0.104662 | 0.014047 |

Fitting the scaler/PCA on noisy or mixed geometry reduced distribution mismatch somewhat but remained inferior to raw Bottom + Inner coordinates. More PCA hyperparameters or alternative representation objectives might behave differently; these results only cover the configurations tested.

## 6. Geometry reconstruction and prediction (Steps 27–31)

Step 27 plotted **actual vs predicted Part1_E** under identical noisy validation inputs for a clean-trained raw model and the 36-feature radial specialist. The specialist recovered a positive trend but predictions remained compressed toward the mean.

Step 28 trained an **ExtraTreesRegressor** denoiser (150 trees, `min_samples_leaf=2`) using **four radial-noisy training views** of Bottom + Inner coordinates as input and the corresponding **clean Bottom + Inner coordinates** as multivariate targets. The existing `bi_clean` Random Forest was held fixed and then evaluated on the denoiser's reconstructed validation coordinates.

| Same noisy validation inputs | Part1_E R² | Part1_E MAE |
| --- | ---: | ---: |
| `bi_clean` on noisy Bottom + Inner | -1.254235 | 0.021212 |
| `bi_radial` directly on noisy Bottom + Inner | 0.507630 | 0.010204 |
| `denoiser` then `bi_clean` | **0.632186** | **0.008694** |

Geometry coordinate MAE for Bottom + Inner: noisy **0.021900** vs reconstructed **0.001923**, a **91.2% reduction**. All reported Step 28 comparisons use the same 1,302 explored validation cases and the same radial noise condition.

Step 29 produced a three-panel prediction scatter plot (original -> radial specialist -> denoiser + clean predictor), saved in Colab as `Part1E_Denoising_Comparison.png`. **The image was shown in chat but is not stored in this repository**.

Step 30 plotted per-simulation Bottom reconstruction error, Inner reconstruction error, and absolute Part1_E prediction error against actual Part1_E. Visual inspection showed relatively steady, small Bottom errors, somewhat higher Inner errors, and a **U-shaped prediction error**, smallest around 0.12–0.14 and increasing at the low/high modulus extremes. The numerical per-group reconstruction MAE and error-correlation outputs were **not supplied**; do not infer exact correlations from the plots.

Step 31 compared the **same clean-trained Bottom + Inner predictor** receiving perfectly clean geometry vs reconstructed geometry, grouped by actual Part1_E range. The clean-input errors remained small across ranges, while reconstructed-input errors rose sharply near 0.10–0.11 and 0.15–0.16 (as seen in the supplied chart). This identifies **the reconstructed input / clean-model interface as the main observed performance gap**; it does not by itself establish whether the root cause is denoiser bias, information loss, or predictor sensitivity to small reconstruction artifacts.

### Observed bottleneck

Prediction compression remains: small actual moduli tend to be overpredicted and large actual moduli underpredicted. High average geometric reconstruction accuracy alone does not recover every target-relevant feature. Further diagnosis should measure per-feature and per-region distortion, signed prediction residuals, and correlations using the exact saved outputs rather than visual estimates.

## 7. Proposed next experiment (not yet completed)

**Step 32:** train a `RandomForestRegressor` predictor on **out-of-fold reconstructed training geometry** (five-fold cross-fitting of denoisers over unique training simulation rows). Use the fully trained Step 28 denoiser only for validation reconstruction. Compare the reconstruction-aware predictor against frozen `bi_radial` and `bi_clean(denoised)` on the same validation rows. This tests whether a predictor adapted to reconstruction artifacts can improve beyond R² 0.632186. **No Step 32 metrics exist yet.**

Suggested follow-up if Step 32 fails: target-aware denoising or joint reconstruction-plus-Part1_E training, careful comparison of Inner vs Bottom regional reconstruction, multiple noise intensities and random seeds, and uncertainty/calibration at modulus extremes. Do not generate additional FEA simulations without authorization.

## 8. Scientific cautions and reproducibility

1. **Exploratory selection:** Repeated use of one validation split for method selection risks optimistic selection bias. The earlier 1,302-row test was inspected in Steps 3–7. Final performance claims require an independent simulation cohort, or a predeclared/nested evaluation approach with sample families separated where applicable.
2. **No leaked paired samples:** Noise augmentation should stay confined to training. When cross-fitting denoisers, split by **original simulation row** so different corrupted copies of one case never cross train/held-out folds.
3. **Dataset scope:** The current simulation dataset represents **one tissue configuration** despite multiple output columns. This report studies Part1_E only and does not demonstrate specialists for two-, three-, or more-tissue configurations.
4. **Magnitude and seed dependence:** Radial 0.05 and Gaussian 0.05 are not automatically equivalent in realized geometry displacement. Test across seeds and noise magnitudes; report the bottom-jitter contribution separately.
5. **Reconstruction objective:** Absolute coordinate MAE is not necessarily aligned with downstream Part1_E loss. Report both, plus subgroup and modulus-range diagnostics.
6. **Artifact reproducibility:** The active Colab notebook `04_Shape_Analysis_Comparison.ipynb`, fitted models, CSV simulation data, and figure PNGs are **not included** in this documentation commit. The report transcribes metrics the user supplied; it is not a fresh execution of the experiments. Avoid claiming reproducible runs from source alone until the notebook and model/config provenance are archived.
7. **Prior notebooks:** Results in `02_One_Tissue_MoE_.ipynb` and `03_MultiTissue_Orchestrator.ipynb` use different models/splits or tasks and are logged separately in [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md). Do not combine their validation metrics directly with notebook 04.

## 9. Current conclusion

**Bottom + Inner geometry is sufficient for high-accuracy clean prediction under the tested setup.** Radial noise destroys much of that predictive accuracy. Separate PCA helps clean data but neither joint PCA, tested nonlinear Inner embeddings, nor stable global contour descriptors solve full-noise robustness. **Noise-trained models help, and supervised geometry reconstruction followed by the clean-trained predictor is currently the strongest explored notebook-04 strategy** (R² **0.632186**, MAE **0.008694** on this particular noisy validation split). The next immediate question is whether a reconstruction-aware downstream model can overcome the remaining modulus-range bias.
