Spatial Exante Framework for Targeting Sowing Dates using Causal ML and
Policy Learning: Eastern India Example
================
Maxwell Mkondiwa

- <a href="#introduction" id="toc-introduction"><span
  class="toc-section-number">1</span> Introduction</a>
- <a href="#preliminaries" id="toc-preliminaries"><span
  class="toc-section-number">2</span> Preliminaries</a>
  - <a href="#graphics" id="toc-graphics"><span
    class="toc-section-number">2.1</span> Graphics</a>
  - <a href="#descriptives" id="toc-descriptives"><span
    class="toc-section-number">2.2</span> Descriptives</a>
- <a href="#causal-random-forest-model"
  id="toc-causal-random-forest-model"><span
  class="toc-section-number">3</span> Causal Random Forest Model</a>
- <a href="#policy-learning-algorithm-for-treatment-assignment"
  id="toc-policy-learning-algorithm-for-treatment-assignment"><span
  class="toc-section-number">4</span> Policy learning algorithm for
  treatment assignment</a>
- <a href="#distributional-analysis"
  id="toc-distributional-analysis"><span
  class="toc-section-number">5</span> Distributional analysis</a>
- <a href="#transition-matrix-of-the-policy-change"
  id="toc-transition-matrix-of-the-policy-change"><span
  class="toc-section-number">6</span> Transition matrix of the policy
  change</a>

# Introduction

In this notebook, I use a causal machine learning estimator, i.e.,
multi-armed causal random forest with augmented inverse propensity score
weights (Athey et al 2019), to estimate conditional average treatment
effects (CATES) for agronomic practices. These CATEs are estimated for
each individual farm thereby providing personalized estimates of the
potential effectiveness of the practices. I then use a debiased robust
estimator in a policy tree optimization (Athey and Wager 2021) to
generate optimal recommendations in the form of agronomic practices that
maximize potential yield gains.

# Preliminaries

``` r
load("LDS_Public_Workspace.RData")

LDSestim=LDS
table(LDSestim$Sowing_Date_Schedule) 
```


    T5_16Dec T4_15Dec T3_30Nov T2_20Nov T1_10Nov 
         665     1696     3167     1704      416 

## Graphics

``` r
# Bar graphs showing percentage of farmers adopting these practices  

library(tidyverse) 
library(ggplot2)  

bar_chart=function(dat,var){   dat|>     drop_na({{var}})|>     mutate({{var}}:=factor({{var}})|>fct_infreq())|>     ggplot()+     geom_bar(aes(y={{var}}),fill="dodgerblue4")+     theme_minimal(base_size = 16) }   

sow_plot=bar_chart(LDSestim,Sowing_Date_Schedule)+labs(y="Sowing dates") 
 
sow_plot
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-2-1.png)

``` r
library(ggpubr) 
library(tidyverse) 

#Sowing dates 

SowingDate_Options_Errorplot=   LDSestim%>%   
  drop_na(Sowing_Date_Schedule) %>%   
  ggerrorplot(x = "Sowing_Date_Schedule", y = "L.tonPerHectare",add = "mean", error.plot = "errorbar", color="steelblue", ggtheme=theme_bw())+   
  labs(x="Sowing date options",y="Wheat yield (t/ha)")+   
  theme_minimal(base_size = 16)+
  coord_flip()  
```

    Warning: The `fun.y` argument of `stat_summary()` is deprecated as of ggplot2 3.3.0.
    i Please use the `fun` argument instead.
    i The deprecated feature was likely used in the ggpubr package.
      Please report the issue at <https://github.com/kassambara/ggpubr/issues>.

    Warning: The `fun.ymin` argument of `stat_summary()` is deprecated as of ggplot2 3.3.0.
    i Please use the `fun.min` argument instead.
    i The deprecated feature was likely used in the ggpubr package.
      Please report the issue at <https://github.com/kassambara/ggpubr/issues>.

    Warning: The `fun.ymax` argument of `stat_summary()` is deprecated as of ggplot2 3.3.0.
    i Please use the `fun.max` argument instead.
    i The deprecated feature was likely used in the ggpubr package.
      Please report the issue at <https://github.com/kassambara/ggpubr/issues>.

``` r
SowingDate_Options_Errorplot 
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-2-2.png)

## Descriptives

``` r
setwd("D:/OneDrive/CIMMYT/Papers/IO5.3.1.CropResponseModels/WheatResponse/Policytree_Models") 

library(fBasics)

summ_stats <- fBasics::basicStats(LDSestim[,c("L.tonPerHectare","G.q5305_irrigTimes","Nperha","P2O5perha","Weedmanaged","variety_type_NMWV","Sowing_Date_Early")]) 

summ_stats <- as.data.frame(t(summ_stats)) 

# Rename some of the columns for convenience 

summ_stats <- summ_stats[c("Mean", "Stdev", "Minimum", "1. Quartile", "Median",  "3. Quartile", "Maximum")] %>%   
rename("Lower quartile" = '1. Quartile', "Upper quartile"= "3. Quartile")  

summ_stats 
```

                             Mean     Stdev Minimum Lower quartile    Median
    L.tonPerHectare      2.990635  0.854990     0.2         2.4000   3.00000
    G.q5305_irrigTimes   2.286183  0.767177     1.0         2.0000   2.00000
    Nperha             130.220556 37.038394     0.0       105.1852 132.54321
    P2O5perha           59.038865 19.629879     0.0        45.4321  59.77908
    Weedmanaged          0.758990  0.427724     0.0         1.0000   1.00000
    variety_type_NMWV    0.530097  0.499126     0.0         0.0000   1.00000
    Sowing_Date_Early    0.277197  0.447644     0.0         0.0000   0.00000
                       Upper quartile  Maximum
    L.tonPerHectare           3.43000   6.5000
    G.q5305_irrigTimes        3.00000   5.0000
    Nperha                  156.00000 298.4691
    P2O5perha                72.69136 212.9630
    Weedmanaged               1.00000   1.0000
    variety_type_NMWV         1.00000   1.0000
    Sowing_Date_Early         1.00000   1.0000

# Causal Random Forest Model

``` r
library(grf)
library(policytree)

LDSestim_sow=subset(LDSestim, select=c("Sowing_Date_Schedule","L.tonPerHectare","I.q5505_weedSeverity_num","I.q5509_diseaseSeverity_num","I.q5506_insectSeverity_num","I.q5502_droughtSeverity_num",                                       "Nperha","P2O5perha","variety_type_NMWV","G.q5305_irrigTimes","A.q111_fGenderdum","Weedmanaged","temp","precip","wc2.1_30s_elev","M.q708_marketDistance","nitrogen_0.5cm","sand_0.5cm", "soc_5.15cm","O.largestPlotGPS.Latitude","O.largestPlotGPS.Longitude"))

library(tidyr)
LDSestim_sow=LDSestim_sow %>% drop_na()


Y_cf_sowing=as.vector(LDSestim_sow$L.tonPerHectare)
## Causal random forest -----------------

X_cf_sowing=subset(LDSestim_sow, select=c("I.q5505_weedSeverity_num","I.q5509_diseaseSeverity_num","I.q5506_insectSeverity_num",
                                                  "Nperha","P2O5perha","variety_type_NMWV","G.q5305_irrigTimes","A.q111_fGenderdum","Weedmanaged","temp","precip","wc2.1_30s_elev",
                                                       "M.q708_marketDistance","nitrogen_0.5cm","sand_0.5cm", "soc_5.15cm","O.largestPlotGPS.Latitude","O.largestPlotGPS.Longitude"))


W_cf_sowing <- as.factor(LDSestim_sow$Sowing_Date_Schedule)

W.multi_sowing.forest <- probability_forest(X_cf_sowing, W_cf_sowing,
  equalize.cluster.weights = FALSE,
  seed = 2
)
W.hat.multi.all_sowing <- predict(W.multi_sowing.forest, estimate.variance = TRUE)$predictions



Y.multi_sowing.forest <- regression_forest(X_cf_sowing, Y_cf_sowing,
  equalize.cluster.weights = FALSE,
  seed = 2
)

print(Y.multi_sowing.forest)
```

    GRF forest object of type regression_forest 
    Number of trees: 2000 
    Number of training samples: 7562 
    Variable importance: 
        1     2     3     4     5     6     7     8     9    10    11    12    13 
    0.000 0.000 0.001 0.021 0.015 0.651 0.193 0.000 0.064 0.001 0.001 0.006 0.001 
       14    15    16    17    18 
    0.001 0.004 0.002 0.023 0.015 

``` r
varimp.multi_sowing <- variable_importance(Y.multi_sowing.forest)
Y.hat.multi.all_sowing <- predict(Y.multi_sowing.forest, estimate.variance = TRUE)$predictions



multi_sowing.forest <- multi_arm_causal_forest(X = X_cf_sowing, Y = Y_cf_sowing, W = W_cf_sowing ,W.hat=W.hat.multi.all_sowing,Y.hat=Y.hat.multi.all_sowing,seed=2) 

varimp.multi_sowing_cf <- variable_importance(multi_sowing.forest)

multi_sowing_ate=average_treatment_effect(multi_sowing.forest, method="AIPW")
```

    Warning in get_scores.multi_arm_causal_forest(forest, subset = subset, debiasing.weights = debiasing.weights, : Estimated treatment propensities take values very close to 0 or 1 meaning some estimates may not be well identified. In particular, the minimum propensity estimates for each arm is
    T5_16Dec: 0 T4_15Dec: 0.006 T3_30Nov: 0.08 T2_20Nov: 0.006 T1_10Nov: 0
    and the maximum is
    T5_16Dec: 0.486 T4_15Dec: 0.799 T3_30Nov: 0.78 T2_20Nov: 0.585 T1_10Nov: 0.544.

``` r
multi_sowing_ate
```

                         estimate    std.err            contrast outcome
    T4_15Dec - T5_16Dec 0.2358984 0.02577961 T4_15Dec - T5_16Dec     Y.1
    T3_30Nov - T5_16Dec 0.4245938 0.02284938 T3_30Nov - T5_16Dec     Y.1
    T2_20Nov - T5_16Dec 0.5638900 0.02572197 T2_20Nov - T5_16Dec     Y.1
    T1_10Nov - T5_16Dec 0.6981874 0.03905789 T1_10Nov - T5_16Dec     Y.1

``` r
varimp.multi_sowing_cf <- variable_importance(multi_sowing.forest)
vars_sowing=c("I.q5505_weedSeverity_num","I.q5509_diseaseSeverity_num","I.q5506_insectSeverity_num",
                                                  "Nperha","P2O5perha","variety_type_NMWV","G.q5305_irrigTimes","A.q111_fGenderdum","Weedmanaged","temp","precip","wc2.1_30s_elev",
                                                       "M.q708_marketDistance","nitrogen_0.5cm","sand_0.5cm", "soc_5.15cm","O.largestPlotGPS.Latitude","O.largestPlotGPS.Longitude")

## variable importance plot ----------------------------------------------------
varimpvars_sowing=as.data.frame(cbind(varimp.multi_sowing_cf,vars_sowing))
names(varimpvars_sowing)[1]="Variableimportance_sowing"
varimpvars_sowing$Variableimportance_sowing=formatC(varimpvars_sowing$Variableimportance_sowing, digits = 2, format = "f")
varimpvars_sowing$Variableimportance_sowing=as.numeric(varimpvars_sowing$Variableimportance_sowing)
varimpplotRF_sowing=ggplot(varimpvars_sowing,aes(x=reorder(vars_sowing,Variableimportance_sowing),y=Variableimportance_sowing))+
   geom_jitter(color="steelblue")+
   coord_flip()+
   labs(x="Variables",y="Variable importance")
 previous_theme <- theme_set(theme_bw(base_size = 16))
 varimpplotRF_sowing
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-4-1.png)

``` r
# Policy tree --------------------------------------
DR.scores_sowing <- double_robust_scores(multi_sowing.forest)

tr_sowing <- policy_tree(X_cf_sowing, DR.scores_sowing, depth = 2) 
plot(tr_sowing)
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-4-2.png)

``` r
tr_sowing3 <- hybrid_policy_tree(X_cf_sowing, DR.scores_sowing, depth = 3) 
tr_sowing3
```

    policy_tree object 
    Tree depth:  3 
    Actions:  1: T5_16Dec 2: T4_15Dec 3: T3_30Nov 4: T2_20Nov 5: T1_10Nov 
    Variable splits: 
    (1) split_variable: O.largestPlotGPS.Latitude  split_value: 25.84 
      (2) split_variable: P2O5perha  split_value: 62.4691 
        (4) split_variable: O.largestPlotGPS.Latitude  split_value: 25.42 
          (8) * action: 5 
          (9) * action: 4 
        (5) split_variable: soc_5.15cm  split_value: 9.1 
          (10) * action: 4 
          (11) * action: 5 
      (3) split_variable: O.largestPlotGPS.Longitude  split_value: 83.47 
        (6) split_variable: precip  split_value: 710.4 
          (12) * action: 5 
          (13) * action: 4 
        (7) split_variable: temp  split_value: 25.6 
          (14) * action: 2 
          (15) * action: 5 

``` r
plot(tr_sowing3)
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-4-3.png)

``` r
tr_sowing4 <- hybrid_policy_tree(X_cf_sowing, DR.scores_sowing, depth = 4) 
tr_sowing4
```

    policy_tree object 
    Tree depth:  4 
    Actions:  1: T5_16Dec 2: T4_15Dec 3: T3_30Nov 4: T2_20Nov 5: T1_10Nov 
    Variable splits: 
    (1) split_variable: O.largestPlotGPS.Latitude  split_value: 25.84 
      (2) split_variable: P2O5perha  split_value: 62.4691 
        (4) split_variable: soc_5.15cm  split_value: 16.2 
          (8) split_variable: O.largestPlotGPS.Latitude  split_value: 25.42 
            (16) * action: 5 
            (17) * action: 4 
          (9) split_variable: I.q5509_diseaseSeverity_num  split_value: 2 
            (18) * action: 4 
            (19) * action: 5 
        (5) split_variable: I.q5505_weedSeverity_num  split_value: 3 
          (10) split_variable: soc_5.15cm  split_value: 9.1 
            (20) * action: 4 
            (21) * action: 5 
          (11) split_variable: precip  split_value: 785.4 
            (22) * action: 4 
            (23) * action: 5 
      (3) split_variable: O.largestPlotGPS.Longitude  split_value: 83.47 
        (6) split_variable: P2O5perha  split_value: 56.7901 
          (12) split_variable: precip  split_value: 710.4 
            (24) * action: 5 
            (25) * action: 4 
          (13) split_variable: temp  split_value: 26.05 
            (26) * action: 5 
            (27) * action: 4 
        (7) split_variable: P2O5perha  split_value: 74.963 
          (14) split_variable: temp  split_value: 25.6 
            (28) * action: 2 
            (29) * action: 5 
          (15) split_variable: temp  split_value: 26.1333 
            (30) * action: 5 
            (31) * action: 4 

``` r
plot(tr_sowing4)
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-4-4.png)

``` r
tr_assignment_sowing=LDSestim_sow

tr_assignment_sowing$depth2 <- predict(tr_sowing, X_cf_sowing)
table(tr_assignment_sowing$depth2)
```


       4    5 
    2031 5531 

``` r
tr_assignment_sowing$depth3 <- predict(tr_sowing3, X_cf_sowing)
table(tr_assignment_sowing$depth3)
```


       2    4    5 
     183 1252 6127 

``` r
tr_assignment_sowing$depth4 <- predict(tr_sowing4, X_cf_sowing)
table(tr_assignment_sowing$depth4)
```


       2    4    5 
     159 1565 5838 

# Policy learning algorithm for treatment assignment

``` r
library(rgdal)

tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2==1]="T5_16Dec"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2==2]="T4_15Dec"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2==3]="T3_30Nov"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2==4]="T2_20Nov"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2==5]="T1_10Nov"

tr_assignment_sowingsp= SpatialPointsDataFrame(cbind(tr_assignment_sowing$O.largestPlotGPS.Longitude,tr_assignment_sowing$O.largestPlotGPS.Latitude),data=tr_assignment_sowing,proj4string=CRS("+proj=longlat +datum=WGS84"))

library(mapview)
mapviewOptions(fgb = FALSE)
tr_assignment_sowingspmapview=mapview(tr_assignment_sowingsp,zcol="depth2_cat",layer.name="Recommended sowing dates")
tr_assignment_sowingspmapview
```

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-5-1.png)

# Distributional analysis

``` r
library(ggridges)
library(dplyr)
tau.multi_sowing.forest=predict(multi_sowing.forest, target.sample = "all",estimate.variance=TRUE)

tau.multi_sowing.forest=as.data.frame(tau.multi_sowing.forest)


tau.multi_sowing.forest_X=data.frame(LDSestim_sow,tau.multi_sowing.forest)


# Ridges -------------------
tau.multi_sowing.forest_pred=tau.multi_sowing.forest[,1:4]

library(dplyr)
library(reshape2)
tau.multi_sowing.forest_pred=rename(tau.multi_sowing.forest_pred,"T4_15Dec - T5_16Dec"="predictions.T4_15Dec...T5_16Dec.Y.1")

tau.multi_sowing.forest_pred=rename(tau.multi_sowing.forest_pred,"T3_30Nov-T5_16Dec"="predictions.T3_30Nov...T5_16Dec.Y.1")

tau.multi_sowing.forest_pred=rename(tau.multi_sowing.forest_pred,"T2_20Nov-T5_16Dec"="predictions.T2_20Nov...T5_16Dec.Y.1")

tau.multi_sowing.forest_pred=rename(tau.multi_sowing.forest_pred,"T1_10Nov-T5_16Dec"="predictions.T1_10Nov...T5_16Dec.Y.1")


tau.multi_sowing.forest_pred_long=reshape2::melt(tau.multi_sowing.forest_pred[,1:4])

ggplot(tau.multi_sowing.forest_pred_long, aes(x=value, y=variable, fill = factor(stat(quantile)))) +
  stat_density_ridges(
    geom = "density_ridges_gradient", calc_ecdf = TRUE,
    quantiles = 4, quantile_lines = TRUE
  ) +
  scale_fill_viridis_d(name = "Quartiles")+
  theme_bw(base_size = 16)+labs(x="Wheat yield gain(t/ha)",y="Sowing date options")
```

    Warning: `stat(quantile)` was deprecated in ggplot2 3.4.0.
    i Please use `after_stat(quantile)` instead.

    Warning: Using the `size` aesthetic with geom_segment was deprecated in ggplot2 3.4.0.
    i Please use the `linewidth` aesthetic instead.

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-6-1.png)

# Transition matrix of the policy change

``` r
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2 == 1] <- "T5_16Dec"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2 == 2] <- "T4_15Dec"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2 == 3] <- "T3_30Nov"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2 == 4] <- "T2_20Nov"
tr_assignment_sowing$depth2_cat[tr_assignment_sowing$depth2 == 5] <- "T1_10Nov"


library(ggalluvial)
library(data.table)
tr_assignment_sowingDT = data.table(tr_assignment_sowing)
TransitionMatrix_sowing <- tr_assignment_sowingDT[, (sum <- .N), by = c("Sowing_Date_Schedule", "depth2_cat")]
library(dplyr)
TransitionMatrix_sowing <- rename(TransitionMatrix_sowing, Freq = V1)

library(scales)
transitionmatrixplot_sowing <- ggplot(
    data = TransitionMatrix_sowing,
    aes(axis1 = Sowing_Date_Schedule, axis2 = depth2_cat, y = Freq)
) +
    geom_alluvium(aes(fill = depth2_cat)) +
    geom_stratum() +
    # geom_text(stat="stratum", aes(label=after_stat(stratum),nudge_y =5))+
    geom_text(stat = "stratum", aes(label = paste(after_stat(stratum), percent(after_stat(prop))))) +
    scale_x_discrete(
        limits = c("Sowing_Date_Schedule", "depth2_cat"),
        expand = c(0.15, 0.05)
    ) +
    scale_fill_viridis_d() +
    theme_void(base_size = 20) +
    theme(legend.position = "none")

transitionmatrixplot_sowing
```

    Warning in to_lodes_form(data = data, axes = axis_ind, discern =
    params$discern): Some strata appear at multiple axes.

    Warning in to_lodes_form(data = data, axes = axis_ind, discern =
    params$discern): Some strata appear at multiple axes.

    Warning in to_lodes_form(data = data, axes = axis_ind, discern =
    params$discern): Some strata appear at multiple axes.

    Warning: Using the `size` aesthetic in this geom was deprecated in ggplot2 3.4.0.
    i Please use `linewidth` in the `default_aes` field and elsewhere instead.

![](Causal_RF_policy_learning_wheat_sowing_public_files/figure-commonmark/unnamed-chunk-7-1.png)
