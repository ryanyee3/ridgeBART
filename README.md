# ridgeBART

## Motivation
The **ridgeBART** package is a computationally efficient alternative to treed Gaussian process models.

## Installation and basic usage

The package source files are contained in the sub-directory `ridgeBART`.
To install, download that directory and build and install from the command line:
```
R CMD build ridgeBART
R CMD INSTALL ridgeBART_1.0.0.tar.gz
```

Alternatively, you can install using `devtools::install_github()`:
```
devtools::install_github(repo = "ryanyee3/ridgeBART", subdir = "ridgeBART")
```
