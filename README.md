<img src="./docs//images/bk_name_en.png" alt="Project Banner" width="100%" />

---

<div align="center">

# Specialized Project

# Thunderstorm Nowcasting  Geometry & Topology Approach

</div>

---

<div align="center">

## Team Members

| Full Name       | Student ID |
| --------------- | ---------- |
| Tran Quoc Hieu  | 2252217    |
| Tran Quoc Trung | 2252859    |
| Nguyen Anh Khoa | 2252352    |

</div>

---
## Table of Contents

- [Team Members](#team-members)
- [Project Scope](#project-scope)
- [1. Project Architecture](#1-project-architecture)
- [2. Implementation - Core Idea](#2-implementation---core-idea)
  - [2.1 Stage 0: Image Preprocessing](#21-stage-0-image-preprocessing)
  - [2.2 Stage 1: Identification](#22-stage-1-identification)
  - [2.3 Stage 2: Tracking](#23-stage-2-tracking)
  - [2.4 Stage 3: Nowcasting](#24-stage-3-nowcasting)
- [3. Tracking Analysis](#3-tracking-analysis)
- [4. Framework Details](#4-framework-details)
  - [4.1 Identification Framework](#41-identification-framework)
  - [4.2 Storm Representation](#42-storm-representation)
  - [4.3 Tracking Framework](#43-tracking-framework)
  - [4.4 Nowcasting Framework](#44-nowcasting-framework)

---


## Project Scope

This project implements a radar-based weather nowcasting system for thunderstorms using topology approaches and Computer Vision (CV) and Image Processing techniques rather than relying on Deep Learning. The system implements a Identification, Tracking, and Nowcasting framework. 

**Key Objectives:**
- Identifying convective cells from radar reflectivity (dBZ) images
- Tracking storm evolution across continuous radar scans
- Predicting future storm positions.
- Evaluating prediction accuracy using multiple metrics (POD, FAR, CSI)
- Comparing performance across different tracking and identification algorithms

---

# 1. Project Architecture
```
.
├── app/                                    # UI application components
│   ├── main.py                            # Application entry point
│   ├── run_app.py                         # App runner
│   ├── components/                        # UI components
│   │   ├── sidebar.py
│   │   └── contours_window.py
│   ├── config/                            # Configuration management
│   │   ├── source_config.py
│   │   └── app_config/
│   ├── cores/                             # Core processing
│   │   └── data_processor.py
│   ├── pages/                             # Page definitions
│   │   └── home_page.py
│   └── utils/                             # Utility functions
│       ├── draw_contours.py
│       └── tracking_matching.py
├── src/                                    # Source code for algorithms
│   ├── cores/                             # Core schemas and metrics
│   │   ├── base/                          # Base classes for storms
│   │   ├── metrics/                       # Evaluation metrics
│   │   ├── polar_description_vector/      # Shape vector implementation
│   │   └── similarity/                    # Similarity measures
│   ├── identification/                    # Storm identification algorithms
│   ├── models/                            # Algorithm implementations
│   │   ├── ata_2021/                      # Adaptive Tracking (ATA)
│   │   ├── etitan_2009/                   # Enhanced TITAN
│   │   ├── titan_1993/                    # TITAN 1993
│   │   ├── ours/                          # Our proposed methods
│   │   └── simple/                        # Simple baseline
│   ├── nowcasting/                        # Nowcasting/forecasting
│   ├── preprocessing/                     # Data preprocessing
│   └── tracking/                          # Tracking algorithms
├── experimental_notebooks/                # Experimental Jupyter notebooks
├── output/                                # Output results and CSVs
├── utils/                                 # Utility functions
│   └── visualization/                     # Visualization utilities
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project configuration
└── README.md                              # This file
```
# 2. Implementation - Core Idea

## 2.1 Stage 0: Image Preprocessing
In this stage, we implement various preprocessing functions, each receives a `frame` from a data source. For example: `images` for `Windy` dataset, or a `dBZ` grid for `NEXRAD` data.
- **Input:** a `frame` from directory data (predefined directory path).
- **Output:** standardized reflectivity radar data 2D grid, where each cell directly indicate dBZ values.
- Source code path: `src/preprocessing/background_preprocessing.py`

**Base class design:** input is path of the `frame`; output is the dBZ maps *(grid)*, with the type `np.ndarray`.

## 2.2 Stage 1: Identification
In this stage, from a single standardized `frame` after preprocessing, we implement source code from different papers to define set of storms in each `frame`.
- **Input:** a dBZ image saved in `np.ndarray` type.
- **Output:** list of contours that represent the contours of each single storms in that dBZ map. We provide some implementation:
    - `SimpleContourIdentifier`: simply identify storms as contiguous pixels of high dBZ values (> **threshold**)
    - `HypothesisIdentifier` (*From [Australia. 2025]*): first process contiguous region and use dilation from maximum center to iteratively process subcells.
    - ...
- Source code for reference: folders `src/identification/`

## 2.3 Stage 2: Tracking
In this stage, after identifying storms in consecutive images, we match storms between two `frames`. We construct a **disparity matrix** that assigns similarity/cost scores between storms from consecutive frames, enabling optimal matching.

**Key approaches:**
- **Hungarian Matching:** Uses the Hungarian algorithm to optimize the total cost of matching storms from consecutive frames.
- **Disparity Analysis:** Uses disparity values to detect storm behavior changes such as splitting (one storm divides into multiple) and merging (multiple storms combine into one).

**Base classes:**
- `BaseMatcher`: Constructs the disparity matrix and performs optimal matching
- `BaseTracker`: Manages the overall tracking pipeline across multiple frames

- Source code for reference: folder `src/tracking/`

## 2.4 Stage 3: Nowcasting
Use history of matching, build a nowcasting class that manage the movement history of storms. This class is also customized to make a prediction on the next position of the storm.

### Evaluation metrics
1. **Probability of Detection (POD):** area of overlapping between predicted and grouth truth over total area of grouth truth. High POD means model detect region of convective cells well.
2. **False Alarm Ratio (FAR):** total area that system predict high dBZ but grouth truth have low dBZ value.
3. **Critical Success Index (CSI):** computed as the ratio of correctly predicted storms to the total area of predicted and true storms minus the area of correctly predicted storms.



# 4. Framework Details

This section describes the complete `identification - tracking - nowcasting` framework with detailed implementations of different algorithms and their core components.

## 4.1 Identification Framework

### Overview
The identification stage extracts storm regions from preprocessed data converted into `frames`. The `frames` are then analyzed, and the output is a list of storms map, each correspond to a timeframe and contains a list of identified storms.

### Implemented Algorithms

- **SimpleIdentifier** (`src/identification/simple_identify.py`): *TITAN_1993*
  - Basic thresholding approach for storm detection
  
- **MorphIdentifier** (`src/identification/morphology__identify.py`): *ETITAN_2007*
  - Uses morphological operations (dilation, erosion) for storm detection
  
- **HypothesisIdentifier** (`src/identification/hypothesis_identify.py`): *AINT_2025*
  - Advanced hypothesis-based detection with iterative subcell processing
  - Processes contiguous regions and applies dilation from maximum centers
  
- **ClusterIdentifier** (`src/identification/cluster_identify.py`): Clustering-based approach
  - Uses clustering techniques for storm region identification

### Common Parameters
All identifiers share these hyperparameters:
- `THRESHOLD`: dBZ threshold for storm detection (high reflectivity patches will be treated as a storm)
- `AREA_FILTER`: Minimum storm area threshold (storms below this are discarded)

## 4.2 Storm Representation

### Storm Object Classes
After identification, storms are stored in class hierarchies inheriting from `BaseStormObject` (`src/cores/base/base_object.py`):

- **StormObject (base class)**: Core attributes: `id`, `contour`
  
- **CentroidStorm**: Extends StormObject with `centroid` coordinate
  
- **ShapeStorm**: Extends StormObject with `shape_vectors` for detailed shape representation
  - Tracks particles and shape vector information for advanced analysis

### StormsMap Class
Container class for all storms at a specific time:
- **Attributes:**
  - `time_frame`: Temporal index or timestamp
  - `storms`: List of StormObject instances

## 4.3 Tracking Framework

### Framework Architecture

The tracking stage matches storms across consecutive frames to build storm trajectories. Three core base classes are implemented in `src/models/base/`:

1. **BaseMatcher** (`base.py`):
   - Methods:
     - `_construct_disparity_matrix()`: Builds cost matrix from two storm lists
     - `_hungarian_matching()`: Solves optimal assignment problem
   - Handles one-to-one storm correspondences

2. **TrackingHistory**:
   - Tracks storm evolution over multiple frames
   - Methods:
     - `forecast()`: Predicts next storm position based on history
     - `update()`: Updates track with new storm assignment

3. **BaseTracker** (`base.py`):
   - Orchestrates the complete tracking pipeline
   - Methods:
     - `fit()`: Processes sequence of storm maps and records history
     - `predict()`: Generates predictions for upcoming frames

### Available Model Implementations

Each model package in `src/models/` implements custom matchers and trackers:

- **`ata_2021/`** (Adaptive Tracking): Custom matcher and tracker with adaptive parameters
- **`etitan_2009/`** (Enhanced TITAN): Morphological enhancements to tracking
- **`titan_1993/`** (Original TITAN): Classic storm tracking approach
- **`simple/`**: Baseline implementation with simple cost functions
- **`ours/`**: Custom proposed method with particle matching
- **`ours_new/`**: Improved version of proposed method

### Core Tracking Components
Located in `src/tracking/`:
- `base.py`: Base classes and interfaces
- `hungarian_matching.py`: Hungarian algorithm implementation for optimal matching

## 4.4 Nowcasting Framework

### Forecasting Pipeline
The nowcasting stage predicts future storm positions and characteristics using tracking history. The forecasting API is integrated into the tracking framework since most algorithms require forecasting to work alongside tracking.

### Implementation Strategy
- Uses tracking history to compute storm motion vectors
- Applies velocity-based extrapolation for position prediction
- Supports integration with different tracker implementations
- No separate nowcasting stage needed; predictions are generated through the tracking module

### Core Components
Located in `src/nowcasting/`:
- `base.py`: Base nowcasting interface
- `constant_vector.py`: Constant velocity nowcasting model

