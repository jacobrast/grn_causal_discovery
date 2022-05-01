## Instructions 

To run this code, using an AWS dlami EC2 instance is recommended. This was run using the `pytorch_p37` environment with the addition of the following package:

`pip install pickle5`

To run Baseline GRNULAR, navigate to `original-grnular` and run 

`python main_grnular.py`

Results will be printed on stdout and will be saved as `DS1_results_original.pickle`

To run GRNULAR + TopoDiffVae navigate to `grnular+vae` and run 

`python main_grnular.py`

Results will be printed on stdout and will be saved as `DS1_results_new.pickle`
