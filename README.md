# Moraine age calculation from CRN samples using a Gaussian Fitting approach

This code is part of Stübner, K., Bookhagen, B., Merchel, S., Lachner, J., Gadoev, M. (in review): *Unraveling the Pleistocene glacial history of the Pamir Mountains, Central Asia*

Please cite this publication when using this code.

Contact [Konstanze Stübner](mailto:kstueb@gmail.com?subject=[GitHub]%20MoraineAge_GaussianMixture) or [Bodo Bookhagen](mailto:bodo.bookhagen@uni-potsdam.de?subject=[GitHub]%20MoraineAge_GaussianMixture) with questions.

# Contents

This repository contains a module for fitting Gaussian distributions to moraine ages derived from cosmogenic radionuclide (CRN) dating (e.g., 10Be, 26Al). The steps are described in the accompanying manuscript Stübner et al., (in review) and are detailed in the source code.

# Installation
## Required Python packages
The code is written in python3.
You will need to setup the required python packages. Best to use conda or pip for that. For a command-line conda approach, use the following:

`conda install scipy numpy scikit-learn lmfit pandas`

## Python source code
The main module is `moraine_age_calculator.py`. This is a well document code that defines the class BoulderAges and required functions. If you run it as a standalone program from the command line (`python3 moraine_age_calculator.py`), it will generate some random test data, plot the distributions as graphs and give misfit values on the screen.

# Example

Here we use compiled CRN moraine ages from the literature for the greater Pamir region. The data are stored in the files `data_compilation_0mmky.csv` and `data_compilation_1mmky.csv` - they contain age data calculated with erosion rates of 0 mm/ky and 1 mm/ky and thus represent minimum ages and best-fit estimates.

The CSV file has the following format (Example:):

  ```
  group,groupName,Age,intErr,extErr,sample,reference
  B21,Gu-Sha,35.6091,0.42834,2.64372,13TMS011,"Stuebner et al., 2017"
  ```
Mandatory columns are (in any order):  
`group`: Group or area identifier  
`groupName`: Alias for the group  
`Age`: CRN age (in ky)  
`intErr`: internal Error (in ky)

Additional columns ignored by the code are:  
`extErr`: external Error (in ky)  
`sample`: Sample Name  
`reference`: Reference

The python code `batch_processing.py` reads these files and generate plots for each group. Starting it with  
`python3 batch_processing.py` will process the entire file `data_compilation_0mmky.csv` (this will take a few minutes). It creates a directory called `Out` that contains figures of all groups and an output `report.csv`.

You can specify the input data file and the name of the output directory with the options `-f` and `-o`, e.g.,  
`python3 batch_processing.py -f ./data_compilation_1mmky.csv -o Out_1mmky`.

Run `python3 batch_processing.py --help` for a help message.
