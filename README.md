# README #

##What is this?##

This is a MATLAB toolkit used for the experimentation of the following paper on signature verification. This paper is appeared in ICPR 2016 as an "oral" presentation.

## Paper ##
Anjan Dutta, Umapada Pal and Josep Lladós. "Compact Correlated Features for Writer Independent Signature Verification", In ICPR, Cancún, Mexico, 2016.

## Dependencies ##
1. vlfeat.
2. libsvm.

## Steps to run ##
1. In the 'MatLabCode/main_signature_verification_compcorr.m' file update the following three variables as described:
(ii) dir_libsvm: path to the 'matlab' folder inside libsvm.
(iii) dir_vlfeat: path to the vlfeat root folder.
(iv) Signsroot: path to the containing folder of the README.txt file. 
2. Run the script and it will produce the output with precomputed histograms on CEDAR. So, after execution, it should produce the following output:

Computing kernel for classification...Done.
Cross Validation Accuracy = 99.9758%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Accuracy = 100% (828/828) (classification)
Accuracy = 100.00, EER = 0.00

**This is the result on a subset (30%) of the dataset. Note the percent_dataset = 0.3 at the beginning of the code.**

To compute the results on the whole dataset, set the variable 'percent_dataset' (within the switch-case 'CEDAR', see upper portion of the code) equal to 1.0. It will need more RAM. It should bring the following output:

Computing kernel for classification...Done.
Cross Validation Accuracy = 99.9964%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Cross Validation Accuracy = 100%
Accuracy = 100% (2760/2760) (classification)
Accuracy = 100.00, EER = 0.00

## Note ##

1. Precomputed histograms only on CEDAR dataset is provided with this toolkit, those for GPDS have not been provided as they are big in size.
2. To recompute results, remember to set the variable 'precomputed_histograms' to false.
3. To produce any results on GPDS dataset, put the corresponding images inside the GPDS folder. Getting results on GPDS dataset will take long time.
4. This code has been checked only on Linux platform (Ubuntu 14.04, 16.04). Please report bug to Anjan Dutta by writing an email at duttanjan@gmail.com.

## Licensing ##
Copyright (C) 2016 Anjan Dutta.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.