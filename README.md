# Mono-V-SLAM

- Team Members: 
    - Aswath Muthuselvam(118286204)
    - Akhil  
    - Vishal 
    - Shailesh Pranav Rajendran(11828)
- Date: 04/29/2022
- Course: ENPM673 - Perception for Autonomous Robots


# Folder structure
```bash
$ tree -L 2
.
├── code
│   ├── main.py
│   └── utils
│        ├── read_frame.py
|        ├── bundle_adjust.py
|        ├── loop_closure.py
|        ├── recover_scale.py
|        └── bag_of_words.py
├── dataset
└── README.md
```

# Setup code
## Setup dataset
1. Download dataset from 
2. Organize your `<datatset-folder>` folder in this structure:



## Setup environment
```bash
# ~/.bashrc
export SLAM_DATA_FOLDER = <datatset-folder>
```


# Run the code
- Supervised Learning:
```bash
#Training:
jupyter nbconvert --to python code/supervised_learning/train.ipynb

#Inference:
run code/supervised_learning/open_loop_test.ipynb

```

- Reinforcement Learning:
```bash
#Training:
run code/reinforcement_learning/gym_environment_play_and_learn.ipynb
```

# Outputs
Youtube video of simualtion output is available in this [link]().

