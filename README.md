# Mono-V-SLAM

## Authors 
- Aswath Muthuselvam(118286204)
- Akhilrajan Vethirajan(117431773)  
- Vishal Kanna Sivakumar()
- Shailesh Pranav Rajendran(11828)

    - Date: 04/29/2022
    - Course: ENPM673 - Perception for Autonomous Robots
---


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
python3 code/main.py
```

# Outputs
Youtube video of simualtion output is available in this [link]().

