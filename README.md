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
`docs/requirements.txt` file. 
Run conda `conda create --name <env> --file <this file>`

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

Trained env for `PandaPickAndPlaceAndThrow-v1`:

![Bullet_Physics_PandaPickAndPlaceAndThrowTrained](https://user-images.githubusercontent.com/92969814/221651495-50b2c340-bdf4-4f0f-ac3c-43160bd9f7c5.gif)

Trained env for `PandaPickAndPlaceAndMove-v1`:

![Bullet_Physics_PandaPickAndPlaceAndMoveTrained](https://user-images.githubusercontent.com/92969814/221654873-74b91669-07c7-419e-af76-c18a4cdd8ff6.gif)



### Site note

It takes approximately 1 million episodes, before an agent reach 
convergence. The models were trained on a server with two
Nvidia Quadro 8000. Not every experiment reached convergence.


