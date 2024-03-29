# Custom Panda Gym Environments

Students project at the University of Applied Sciences Dresden (HTW Dresden). The goal
was to customize an existing reinforcement learning problem and to successfully train a model.
The base for this project was the panda-gym repository (https://github.com/qgallouedec/panda-gym).

Models were successfully trained with the TQC algorithm and partly also with the SAC algorithm. We used 
stable-baselines3 and sb3-contrib to integrate the algorithms. 

We have implemented two custom gym environments:
- `PandaPickAndPlaceAndThrow-v1`
- `PandaPickAndPlaceAndMove-v1`

This repository includes trained models for the environments, available as zip files. 
More information at the usage section.

## Installation 

This project was executed in a conda environment. Under Windows we used PyCharm as an execution IDE, 
under Linux (Ubuntu) you can execute the code throw the CLI. 

Following commands were executed:
```bash
conda create -n example_name python=3.10
conda activate name
conda install -c conda-forge stable-baselines3[extra]
conda install -c conda-forge tensorboard
pip install panda-gym==2.0
pip install sb3-contrib
```

Because of the fast developing paste in the reinforcement learning sector the safer way is to use the
`docs/requirements.yaml` file. 
Run conda `conda create --name <env> --file <this file>`

### Step by step installation guide

1. git clone https://github.com/balooox/CustomPandaPickAndPlaceEnv.git
2. cd ./CustomPandaPickAndPlaceEnv
3. use Anaconda:
    1. Under Windows: open the anaconda prompt (anaconda3)
    2. Under Linux: make sure anaconda is integrated into the cli
4. conda env create --file ./docs/environment.yml
5. conda activate custom_panda_env

## Usage

### showcase.py 
To get an overview over the custom environments, you can run the script showcase.py !

```bash
python ./showcase.py env
# For example
python ./showcase.py PandaPickAndPlaceAndThrow-v1
```

### train.py
You can train a model with the train.py script. Available algorithms are TQC and 
SAC ( but you can extend the code and include different algorithms available
through stable-baselines3 and sb3_contrib, like PPO etc... ).

```bash
python ./train env algo amount
# For example
python ./train.py PandaPickAndPlaceAndThrow-v1 TQC 1000000
```

During training (or afterwards) you can visualize the training process by
using tensorboard. To do so, run following command in the CLI: 
```bash
tensorboard --logdir ./tensorboard
```

### enjoy.py
After a training, a zip file is saved under the ./trained directory.
You can run enjoy.py two see a visualized result of the training.

```bash
python ./enjoy env algo file
# For example
python ./enjoy PandaPickAndPlaceAndThrow-v1 TQC ./benchmark/PandaPickAndPlaceAndThrow-v1/TQC/monitor.zip 
```

### evaluate.py
During training, a printed log on the CLI gives you information about the 
success rate and the mean reward for each episode. To truly evaluate a model use the evaluate.py.

```bash
python  ./evaluate env algo file
# For example
python ./evaluate PandaPickAndPlaceAndThrow-v1 TQC ./benchmark/PandaPickAndPlaceAndThrow-v1/TQC/monitor.zip
```

## Environments

This custom environment was used for `PandaPickAndPlaceAndMove-v1` and `PandaPickAndPlaceAndThrow-v1`. 

![PandaPickAndPlaceAndThrow-v1](https://user-images.githubusercontent.com/7521492/221948141-d4e49583-81b7-4336-ac62-279728e0e46f.gif)



### Trained model 

#### Trained env for `PandaPickAndPlaceAndThrow-v1`:          

![PandaPickAndPlaceAndThrow-v1-trained](https://user-images.githubusercontent.com/7521492/221982843-faf17026-16b2-43c0-a2f0-6275f077b3d7.gif)

Mean reward: 
                             
<img src="https://user-images.githubusercontent.com/92969814/221976736-8bb401b0-1bcf-413b-9210-22be2e66df8e.png" data-canonical-src="https://user-images.githubusercontent.com/92969814/221976736-8bb401b0-1bcf-413b-9210-22be2e66df8e.png" width="50%" />

Success rate: 

<img src="https://user-images.githubusercontent.com/92969814/221976917-7b4a0e63-bd2b-4e53-ad4a-4aef5b7cab0d.png" data-canonical-src="https://user-images.githubusercontent.com/92969814/221976917-7b4a0e63-bd2b-4e53-ad4a-4aef5b7cab0d.png" width="50%" />





#### Trained env for `PandaPickAndPlaceAndMove-v1`:

![PandaPickAndPlaceAndMove-v1-trained](https://user-images.githubusercontent.com/7521492/221985066-4f9ba9d9-ab32-4818-a1d1-13ddc745766f.gif)


Mean reward:

<img src="https://user-images.githubusercontent.com/92969814/221974043-54f852b3-4c67-42e1-a42b-86595c7b630a.png" data-canonical-src="https://user-images.githubusercontent.com/92969814/221974043-54f852b3-4c67-42e1-a42b-86595c7b630a.png" width="50%" />

Success rate:

<img src="https://user-images.githubusercontent.com/92969814/221976224-1755d64f-a4a1-47b3-b8ef-b35d676b099b.png" data-canonical-src="https://user-images.githubusercontent.com/92969814/221976224-1755d64f-a4a1-47b3-b8ef-b35d676b099b.png" width="50%" />





### Site note

It takes approximately 1 million episodes, before an agent reach 
convergence. The models were trained on a server with two
Nvidia Quadro 8000. Not every experiment reached convergence.


