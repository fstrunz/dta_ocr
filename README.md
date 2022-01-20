# Automated [DTA](https://deutschestextarchiv.de) OCR Workflow
Scripts are to be executed in this order:
1. `download.py`: Downloads facsimiles.
2. `segment.py`: Preprocesses (binarises, deskews) the facsimiles and segments them using [Kraken](http://kraken.re/master/index.html). 
3. `predict.py`: Runs [Calamari](https://github.com/Calamari-OCR/calamari) on the segmentations to predict their contents.