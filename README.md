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
1. Download `odometry data set (grayscale, 22 GB)`, `odometry ground truth poses (4 MB)`, `odometry data set (calibration files, 1 MB)` dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Organize your `<datatset-folder>` folder in this structure:
```
<dataset-folder>
    ├── poses
    └── sequences
```

## Setup environment
```bash
# ~/.bashrc
export SLAM_DATA_FOLDER = <datatset-folder>
```


# Run the code
- Supervised Learning:
```bash
python3 code/main.py
```

# Outputs
Youtube video of simualtion output is available in this [link]().

