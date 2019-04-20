Below you can find a outline of how to reproduce our solution for the
Men's 2019 March Madness competition.

If you run into any trouble with the setup/code or have any questions
please contact Young at leey634@gmail.com

# HARDWARE
Canonical, Ubuntu, 18.04 LTS, amd64 bionic image build on 2019-02-12 (AWS AMI)

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.6.7

# DATA PROCESSING
python ./prepare_data.py

# MODEL BUILD
python ./train.py

# GENERATE SUBMISSION
python ./predict.py

# REFERENCE
Our development workflow can be found in the repo below. It contains
source files and notebooks for several other methods we've explored
but didn't adopt for the final submission. The project website
documents initial model exploration.

Github: https://github.com/YouHoo0521/kaggle-march-madness-men-2019/

Website: https://youhoo0521.github.io/kaggle-march-madness-men-2019/