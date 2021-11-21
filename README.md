# STEllar Parameter Predictors (STEPPs)

Repository for the development of machine learning models that aim to predict, based on the magnitudes of a star on a certain set of filters, the values of the three main stellar parameters of a star: Effective Temperature (T<sub>eff</sub>), Surface Gravity (logg) and Metalicity ([Fe/H]).

## Data Used

The surveys used on this project were the Javalambre Photometric Local Universe Survey (**J-PLUS**: magnitudes for 12 different filters), the Wide-field Infrared Survey Explorer (**WISE**: magnitudes for 4 different filters) and the Large Sky Area Multi-Object Fibre Spectroscopic Telescope (**LAMOST**: values for Teff, logg and FeH). The description of the 16 filters used can be found on the table below (details taken from the [JPLUS Official Website][1] and the [SVO Filter Profile Service][2]):

| Filter | Survey | Central Wavelength (nm) |   | Filter | Survey | Central Wavelength (nm) |
|:------:|:------:|:-----------------------:|:-:|:------:|:------:|:-----------------------:|
|  uJAVA | J-PLUS |          348.5          |   |  J0660 | J-PLUS |          660.0          |
|  J0378 | J-PLUS |          378.5          |   |  iSDSS | J-PLUS |          766.8          |
|  J0395 | J-PLUS |          395.0          |   |  J0861 | J-PLUS |          861.0          |
|  J0410 | J-PLUS |          410.0          |   |  zSDSS | J-PLUS |          911.4          |
|  J0430 | J-PLUS |          430.0          |   |   W1   |  WISE  |          3352.6         |
|  gSDSS | J-PLUS |          480.3          |   |   W2   |  WISE  |          4602.8         |
|  J0515 | J-PLUS |          515.0          |   |   W3   |  WISE  |         11560.8         |
|  rSDSS | J-PLUS |          625.4          |   |   W4   |  WISE  |         22088.3         |

As well as the 16 magnitudes, we also calculated all the 120 possible combinations (also called colors) between them, and the 136 resulting features were used as input for the models. The initial J-PLUS + WISE sample had 1.340.602 objects with measured magnitude values for all 16 filters. After crossing this sample with LAMOST, we obtained 186.232 objects in common between the two, and this was our working sample to tune, train and test the models.

## Hyperparameter Optimization
We chose to base our predictors on the **[Random Forest][3]** machine learning model, and since many of the 136 input features had little to no valuable information to add to the model, a Recursive Feature Elimination (**[RFE][4]**) was performed on our input data to choose only the best features before passing them to the Random Forest.

However, before any real training or testing, it was necessary to tune and find the best model hyperparameters for each STEPP. In our case, this was done through a 5-fold **[Cross Validation][5]** (repeated 3 times) on 75% of the initial sample, and the hyperparameters that we chose to optimize were:

1. **n_features**: The number of features that the RFE passes to the RF model (Values tested: 10, 15, 45, 60, 136);
2. **max_features**: The fraction of features used by the random forest to perform each split (Values tested: 0.1, 0.25, 0.5, 0.75, 1.0)
3. **n_trees**: The number of trees on the forest (Values tested: 50, 100).
4. **min_samples_leaf**: The minimal number of samples remaining on each side of a split for it to be considered possible (Values tested: 1, 10)

In total, each one of the three STEPPs was tested with 50 different combinations of hyperparameters, and the notebooks used for that can be found inside the [hyperparameter_tuning](hyperparameter_tuning/) folder. Below are the main results, where we considered the R2 score as our main metric to optimize:

### T<sub>eff</sub> Predictor Hyperparameter Tuning
<img align="left" src="hyperparameter_tuning/teff/rf_teff_R2_heatmap.jpg" width=550>

**5 Best models**
|    Combination   |   R2   |
|:----------------:|:------:|
|  (45, 0.25, 100, 1) | 0.9688 |
|  (136, 0.1, 100, 1) | 0.9686 |
|  (60, 0.25, 100, 1) | 0.9686 |
|  (60, 0.5, 100, 1)  | 0.9686 |
| (136, 0.25, 100, 1) | 0.9686 |
<br>
As can be seen on the heatmap, all of the combinations tested resulted in R2 scores above 0.965, and the difference between the best and worst models is very small (around 0.004). Also, as expected, there is almost no difference in the score after n_features = 45 (with the exception of a much greater training time).
<br>
It is interesting to point out that every model with n_trees = 100 performed slightly better than its counterpart with n_trees = 50, and that for a fixed value of n_features, a model with max_features = 0.25 performs better than all the others. Also, all the models witt msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

### logg Predictor Hyperparameter Tuning
<img align="right" src="hyperparameter_tuning/logg/rf_logg_R2_heatmap.jpg" height=550>

**5 Best models**
|    Combination   |   R2   |
|:----------------:|:------:|
|  (60, 0.25, 100, 1) | 0.8294 |
|  (45, 0.25, 100, 1) | 0.8292 |
|  (45, 0.5, 100, 1)  | 0.8284 |
|  (60, 0.5, 100, 1)  | 0.8281 |
| (136, 0.25, 100, 1) | 0.8275 |
<br>
Although the logg predictors performed considerably worse than the T<sub>eff</sub> predictors, taking into consideration that a R2 score of 0.8294 amounts to a correlation of 91.1% between the predicted and real values, their results are still very good. 
<br>
Again, increasing the n_features hyperparameter above 45 brings no real improvement to the models, and in general the models with n_trees = 100 performed better than their counterparts with n_trees = 50. Also, models with max_features = 0.25 were the best performing ones when compared to others with the same value n_features. Also, all the models witt msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

### [Fe/H] Predictor Hyperparameter Tuning
<img align="left" src="hyperparameter_tuning/feh/rf_FeH_R2_heatmap.jpg" height=550>

**5 Best models**
|   Combination   |   R2   |
|:---------------:|:------:|
| (60, 0.25, 100, 1) | 0.8591 |
| (45, 0.25, 100, 1) | 0.8591 |
|  (60, 0.1, 100, 1) | 0.8583 |
|  (60, 0.5, 100, 1) | 0.8579 |
|  (45, 0.5, 100, 1) | 0.8579 |
<br>
The [Fe/H] predictors performed slightly above than the logg predictors, but still considerably below the T<sub>eff</sub> predictors, and a R2 score of 0.8591 amounts to a correlation 92.7% between the predicted and real values (again, a very good result). 
<br>
For a third time, the increase of n_features above 45 brought no improvement to the models, while an increase from n_trees = 50 to n_trees = 100 resulted in a small increase in performance. Also, with the exception of n_features = 10 and 15, models with max_features = 0.25 were the best performing ones when compared to others with the same value of n_features. Also, all the models witt msl = 10 performed worse than their counterparts with msl = 1.
<br><br><br>

## Model Training and Testing
With the best combinations of hyperparameters finnaly found, it was possible to perform the training and testing of the STEPPs, and all the process can be consulted inside the folder [final_models](final_models/). Due to the size of the files necessary to store the final models, they are not available in this repository. To obtain them, the user can run all the cells inside the [rf_best_models](final_models/rf_best_models.ipynb) notebook, or just download them from this [Google Drive Folder](https://drive.google.com/drive/folders/149tXTgS2P0Y3n512TDhDOHyU31lKuWbb?usp=sharing).
<br>
In all three cases, the final model was trained on the same sample used for the hyperparameter tuning (75% of the initial sample), and then tested on the remaining 25% objects.
### T<sub>eff</sub> Final Predictor
For the effective temperature, the best performing model was the one with (n_features = 45, max_features = 0.25, n_trees = 100). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_teff_test_results.jpg" height=290>

|   Metric  |   Value  |
|:---------:|:--------:|
|    MAE    |  66.083  |
|    RMSE   |  92.810  |
| Max Error | 2077.188 |
|     R2    |   0.970  |

<br/><br/><br/><br/><br/>
### logg Final Predictor
For the surface gravity, the best performing model was the one with (n_features = 60, max_features = 0.25, n_trees = 100). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_logg_test_results.jpg" height=300>

|   Metric  | Value |
|:---------:|:-----:|
|    MAE    | 0.134 |
|    RMSE   | 0.203 |
| Max Error | 3.856 |
|     R2    | 0.833 |

<br/><br/><br/><br/><br/>
### [Fe/H] Final Predictor
For the metalicity, the best performing model was the one with (n_features = 60, max_features = 0.25, n_trees = 100). Using this combination of hyperparameters, the results were:

<img align="left" src="final_models/rf_feh_test_results.jpg" height=300>

|   Metric  | Value |
|:---------:|:-----:|
|    MAE    | 0.103 |
|    RMSE   | 0.146 |
| Max Error | 2.210 |
|     R2    | 0.860 |

<br/><br/><br/><br/><br/>

## References

1. [J-PLUS](http://www.j-plus.es)
2. [Random Forests](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf)
4. [Cross-Validation](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_565)
5. [Scikit-learn](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
6. [TensorFlow](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
7. [Numpy](https://www.nature.com/articles/s41586-020-2649-2)
8. [Pandas](https://zenodo.org/record/5574486)
9. [Seaborn](https://joss.theoj.org/papers/10.21105/joss.03021)
10. [Matplotlib](https://ieeexplore.ieee.org/document/4160265)

[1]: <http://www.j-plus.es/survey/instrumentation>
[2]: <http://svo2.cab.inta-csic.es/theory/fps/index.php?id=WISE>
[3]: <https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf>
[4]: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html>
[5]: <https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-39940-9_565>
