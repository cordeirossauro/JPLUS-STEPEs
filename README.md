# STEllar Parameter Estimators (STEPEs)

Repository for the development of machine learning models that aim to predict, based on the magnitudes of a star on a certain set of filters, the values of the three main stellar parameters of a star: Effective Temperature (T<sub>eff</sub>), Surface Gravity (logg) and Metalicity ([Fe/H]).

## Data Used

The surveys used on this project were the Javalambre Photometric Local Universe Survey (**J-PLUS**: magnitudes for 12 different filters), the Wide-field Infrared Survey Explorer (**WISE**: magnitudes for 4 different filters), the GAIA Survey (magnitudes for 3 different filters) and the Large Sky Area Multi-Object Fibre Spectroscopic Telescope (**LAMOST**: values for Teff, logg and FeH). The description of the 16 filters used can be found on the table below (details taken from the [JPLUS Official Website][1] and the [SVO Filter Profile Service][2]):

| Filter | Survey | Central Wavelength (nm) |   | Filter | Survey | Central Wavelength (nm) |
|:------:|:------:|:-----------------------:|:-:|:------:|:------:|:-----------------------:|
|  uJAVA | J-PLUS |          348.5          |   |  J0660 | J-PLUS |          660.0          |
|  J0378 | J-PLUS |          378.5          |   |  iSDSS | J-PLUS |          766.8          |
|  J0395 | J-PLUS |          395.0          |   |  RP    | GAIA   |          773.0          |
|  J0410 | J-PLUS |          410.0          |   |  J0861 | J-PLUS |          861.0          |
|  J0430 | J-PLUS |          430.0          |   |  zSDSS | J-PLUS |          911.4          |
|  gSDSS | J-PLUS |          480.3          |   |   W1   |  WISE  |          3352.6         |
|  BP    | GAIA   |          505.0          |   |   W2   |  WISE  |          4602.8         |
|  J0515 | J-PLUS |          515.0          |   |   W3   |  WISE  |         11560.8         |
|  G     | GAIA   |          623.0          |   |   W4   |  WISE  |         22088.3         |
|  rSDSS | J-PLUS |          625.4          |   |        |        |                         |

As well as the 19 magnitudes, we also calculated all the 171 possible combinations (also called colors) between them, and the 190 resulting features were used as input for the models. The initial J-PLUS + WISE + GAIA sample was obtained using the J-PLUS DR2 [Data Access Website][6] and had 1.340.602 objects with measured magnitude values for all 19 filters (and a mag_err > 0.1 for the J-PLUS magnitudes). After crossing this sample with the full LAMOST sample (downloaded directly from their own [search tool][7]), we obtained 185.248 objects in common between the two.

To guarantee that we were using only the stars with good values of each parameter we filtered out the objects with teff_err > 300, logg_err > 0.2 or feh_err > 0.4, and this final development sample (used to tune, train and test the models) had around 101.516 stars.

## Hyperparameter Optimization
We chose to base our estimators on the **[Random Forest][3]** machine learning model, and since many of the 190 input features had little to no valuable information to add to the model, a Recursive Feature Elimination (**[RFE][4]**) was performed on our input data to choose only the best features before passing them to the Random Forest.

However, before any real training or testing, it was necessary to tune and find the best model hyperparameters for each STEPE. In our case, this was done through a 4-fold **[Cross Validation][5]** (repeated 2 times) on 75% of the initial sample, and the hyperparameters that we chose to optimize were:

1. **n_features**: The number of features that the RFE passes to the RF model (Values tested: 15, 45, 60, 136);
2. **max_features**: The fraction of features used by the random forest to perform each split (Values tested: 0.25, 0.5, 0.75, 1.0);
3. **n_trees**: The number of trees on the forest (Values tested: 50, 100);
4. **min_samples_leaf**: The minimal number of samples remaining on each side of a split for it to be considered valid (Values tested: 1, 10).

In total, each one of the three STEPEs was tested with 64 different combinations of hyperparameters, and the notebooks used for that can be found inside the [hyperparameter_tuning](hyperparameter_tuning/) folder. Below are the main results, where we considered the R2 score as our main metric to optimize:

### T<sub>eff</sub> Estimator Hyperparameter Tuning
<img align="left" src="hyperparameter_tuning/teff/rf_teff_R2_heatmap.jpg" width=500>

**5 Best models**
|     Combination    |        R2          |
|:------------------:|:------------------:|
|  (60, 0.5, 100, 1) | 0.9751 &pm; 0.0005 |
|  (45, 0.5, 100, 1) | 0.9751 &pm; 0.0006 |
| (60, 0.25, 100, 1) | 0.9751 &pm; 0.0003 |
| (60, 0.75, 100, 1) | 0.9750 &pm; 0.0008 |
| (45, 0.75, 100, 1) | 0.9750 &pm; 0.0006 |
<br>
As can be seen on the heatmap, all of the combinations tested resulted in R2 scores above 0.971, and the difference between the best and worst models is very small (around 0.004). Also, as expected, there is almost no difference in the score after n_features = 45 (with the exception of a much greater training time).
<br>
It is interesting to point out that every model with n_trees = 100 performed slightly better than its counterpart with n_trees = 50. Also, all the models with msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

### logg Estimator Hyperparameter Tuning
<img align="right" src="hyperparameter_tuning/logg/rf_logg_R2_heatmap.jpg" height=500>

**5 Best models**
|     Combination    |        R2          |
|:------------------:|:------------------:|
| (45, 0.25, 100, 1) | 0.8848 &pm; 0.0025 |
| (60, 0.25, 100, 1) | 0.8841 &pm; 0.0028 |
|  (45, 0.5, 100, 1) | 0.8838 &pm; 0.0014 |
|  (60, 0.5, 100, 1) | 0.8834 &pm; 0.0024 |
|  (45, 0.25, 50, 1) | 0.8830 &pm; 0.0039 |
<br>
Although the logg STEPEs performed worse than the T<sub>eff</sub> estimators, taking into consideration that a R2 score of 0.8848 amounts to a correlation of 94.1% between the predicted and real values, their results are still very good. 
<br>
Again, increasing the n_features hyperparameter above 45 brings no real improvement to the models, and in general the models with n_trees = 100 performed better than their counterparts with n_trees = 50. Also, all the models with msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

### [Fe/H] Estimator Hyperparameter Tuning
<img align="left" src="hyperparameter_tuning/feh/rf_FeH_R2_heatmap.jpg" height=500>

**5 Best models**
|     Combination     |        R2          |
|:-------------------:|:------------------:|
|  (60, 0.25, 100, 1) | 0.8689 &pm; 0.0015 |
|  (45, 0.25, 100, 1) | 0.8682 &pm; 0.0021 |
| (190, 0.25, 100, 1) | 0.8679 &pm; 0.0036 |
|  (45, 0.5, 100, 1)  | 0.8674 &pm; 0.0037 |
|  (60, 0.25, 50, 1)  | 0.8673 &pm; 0.0040 |
<br>
The [Fe/H] STEPEs were the worst ones, but since a R2 score of 0.8698 amounts to a correlation 93.26% between the predicted and real values, we can also consider it a good estimator. 
<br>
For a third time, an increase from n_trees = 50 to n_trees = 100 resulted in a small increase in performance. Also, again, all the models with msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

## Model Training and Testing
With the best combinations of hyperparameters finnaly found, it was possible to perform the training and testing of the STEPEs, and all the process can be consulted inside the folder [final_models](final_models/). Due to the size of the files necessary to store the final models, they are not available in this repository. To obtain them, the user can run all the cells inside the [rf_best_models](final_models/rf_best_models.ipynb) notebook, or just download them from this [Google Drive Folder](https://drive.google.com/drive/folders/149tXTgS2P0Y3n512TDhDOHyU31lKuWbb?usp=sharing).
<br>
In all three cases, the final model was trained on the same sample used for the hyperparameter tuning (75% of the initial sample), and then tested on the remaining 25% objects.
### T<sub>eff</sub> Final STEPE
For the effective temperature, if we consider both the average value and the standard deviation of the R2 scores, the best performing model was the one with (n_features = 60, max_features = 0.25, n_trees = 100, msl = 1). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_teff_estimator/test_results.jpg" height=290>

|   Metric  |   Value  |
|:---------:|:--------:|
|    MAE    |  65.374  |
|    RMSE   |  86.807  |
| Max Error | 1063.697 |
|     R2    |   0.976  |

<br/><br/><br/><br/><br/>
### logg Final STEPE
For the surface gravity, the best performing model was the one with (n_features = 45, max_features = 0.25, n_trees = 100, msl = 1). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_logg_estimator/test_results.jpg" height=300>

|   Metric  | Value |
|:---------:|:-----:|
|    MAE    | 0.114 |
|    RMSE   | 0.170 |
| Max Error | 2.198 |
|     R2    | 0.889 |

<br/><br/><br/><br/><br/>
### [Fe/H] Final STEPE
For the metalicity, the best performing model was the one with (n_features = 60, max_features = 0.25, n_trees = 100, msl = 1). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_feh_estimator/test_results.jpg" height=300>

|   Metric  | Value |
|:---------:|:-----:|
|    MAE    | 0.091 |
|    RMSE   | 0.124 |
| Max Error | 1.130 |
|     R2    | 0.868 |

<br/><br/><br/><br/><br/>

## References

1. Cenarro, A., Moles, M., Cristóbal-Hornillos, D., Marín-Franch, A., Ederoclite, A., Varela, J., López-Sanjuan, C., Hernández-Monteagudo, C., Angulo, R., Vázquez Ramió, H., & et al. (2019). J-PLUS: The Javalambre Photometric Local Universe Survey. Astronomy & Astrophysics, 622, A176.<br>
2. Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001).
3. Refaeilzadeh P., Tang L., Liu H. (2009) Cross-Validation. In: LIU L., ÖZSU M.T. (eds) Encyclopedia of Database Systems. Springer, Boston, MA.
4. Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay; 12(85):2825−2830, 2011.
5. Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
6. Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020).
7. The pandas development team. (2020). pandas-dev/pandas: Pandas 1.0.3 (v1.0.3). Zenodo. 
8. Michael L. Waskom (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
9. Hunter, J. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.

## Citation
For information on how to cite this work, follow the link: <br>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5716519.svg)](https://doi.org/10.5281/zenodo.5716519)


[1]: <http://www.j-plus.es/survey/instrumentation>
[2]: <http://svo2.cab.inta-csic.es/theory/fps/index.php?id=WISE>
[3]: <https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf>
[4]: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>
[5]: <https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_565>
[6]: <http://archive.cefca.es/catalogues/jplus-dr2>
[7]: <http://dr6.lamost.org/search>
