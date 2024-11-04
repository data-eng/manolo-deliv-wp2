# manolo-deliv-wp2
WP2 Deliverables

## Feature Extraction ##
Feature extraction relies on the **catch22** feature set that consists of a set of 22 (and can be extended to 24) CAnonical Time-series CHaracteristics (catch22) tailored to the dynamics typically encountered in time-series data-mining tasks. 
catch22 resulted from the larger **hctsa** time-series feature set.

More information about catch22 can be found:

- Open access paper:
Lubba et al. [catch22: CAnonical Time-series CHaracteristics](https://link.springer.com/article/10.1007/s10618-019-00647-x) , Data Min Knowl Disc 33, 1821 (2019).
- Resources: [CAnonical Time-series CHaracteristics](https://time-series-features.gitbook.io/catch22)

The hctsa feature set is described in the following papers:

1. B.D. Fulcher and N.S. Jones. [hctsa: A computational framework for automated time-series phenotyping using massive feature extraction](https://www.cell.com/cell-systems/fulltext/S2405-4712%2817%2930438-6). Cell Systems 5, 527 (2017).

2. B.D. Fulcher, M.A. Little, N.S. Jones. [Highly comparative time-series analysis: the empirical structure of time series and their methods](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2013.0048). J. Roy. Soc. Interface 10, 20130048 (2013).

[Highly comparative time-series analysis using hctsa](https://time-series-features.gitbook.io/hctsa-manual), contains resources and information about hctsa and also outlines the steps required to set up and implement hctsa using the [hctsa package](https://github.com/benfulcher/hctsa). 

catch22 is the result of a data-driven pipeline that distilled reduced subsets of the
most useful and complementary features for classification from thousands of initial
candidates, included in the hctsa toolbox.


### Installation ###

```markdown
conda install -c conda-forge pycatch22