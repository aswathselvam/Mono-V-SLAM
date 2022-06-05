# Mono-V-SLAM

## Authors 
- Aswath Muthuselvam(118286204)
- Akhilrajan Vethirajan(117431773)  
- Vishal Kanna Sivakumar()
- Shailesh Pranav Rajendran(118261997)

    - Date: 04/29/2022
    - Course: ENPM673 - Perception for Autonomous Robots
---

## Output
**Trajectory**
---
![Output Image](/assets/output.png)

**Pointcloud**
---
![Pointcloud](/assets/output.gif)

**Simulation Video**
---
Youtube video - [link](https://youtu.be/Nj5nIHlJmwM)

# Folder structure
```bash
$ tree -L 2
.
├── code
│   ├── main.py
│   └── utils
│        ├── load_dataset.py
│        ├── compute_pose.py
|        ├── bundle_adjust.py
|        ├── loop_closure.py
|        ├── recover_scale.py
│        ├── plotting.py
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
- Add the dataset folder location
```bash
# ~/.bashrc
export SLAM_DATA_FOLDER = <datatset-folder>
```

### Install dependencies
- Install python requirements:
    ```bash
    pip install requirements.txt
    ```
- Install Pangolin for visualization from:
https://github.com/stevenlovegrove/Pangolin

# Run the code
```bash
python3 code/main.py
```



