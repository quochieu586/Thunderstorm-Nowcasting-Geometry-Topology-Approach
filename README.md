# About this project

# A. Project architecture
```
-> app                      # Config for UI visualization
-> data                     # Save the data images
-> src
---> cores                  # Save the cores schema and metrics for evaluation 
-----> contours             # Save StormObject and StormsMap for modeling single storm (cell) and list of all storms in a single time frame
-----> metrics              # Evaluate the predicted image vs actual image
-----> polar_description    # One implemention shape vector: using polar shape vector to modeling vectors of points in contours
-----> similarity
---> identification         # Implement process of determining and extracting list of contours from dbz image
---> models                 # Concatenate logic from 3 stages into a single nowcasting model, latter will be hold main logic.
---> nowcasting             # Modeling the movement of `StormObject` to generate new `StormObject`
---> preprocessing          # Hold all logics for preprocessing an image and converting objects
---> tracking               # Constructing disparity matrix for two consecutive `StormsMap`
```

For **identification**, **nowcasting** and **tracking**, model the base function in `base.py` and implement the logic behind in implemented class.

# B. Implementation - Core Idea

## Stage 0: Image preprocessing
In this stage, we build many preprocessing functions, each of which receives kind of images from a single source (list in directory of `data`). For example: `images` and `MRMS_MergedReflectivityComposite_2022061500`.
- **Input:** an image from directory data (predefined directory path).
- **Output:** standardized reflectivity radar data 1D image, where each pixel directly indicate DBZ values.
- Source code path: `src/preprocessing/background_preprocessing.py`

**Base class design:** input is path of the image; output is the DBZ maps, saved in type of `np.ndarray` data.

## Stage 1: Identification
In this stage, from a single standardized radar image of preprocessing, we implement source code from different papers to define set of storms in this map.
- **Input:** a DBZ image saved in `np.ndarray` type.
- **Output:** list of contours that represent the contours of each single storms in that DBZ map. We provide some implementation:
    - `SimpleContourIdentifier`: simply identify storms as contiguous pixels of high DBZ values (> **threshold**)
    - `HypothesisIdentifier` (*From [Australia. 2025]*): first process contiguous region and use dilation from maximum center to iteratively process subcells.
    - ...
- Source code for reference: folders `src/identification/`

## Stage 2: Tracking
In this stage, after identifying storms in consecutive images, we matched storms between two scans. About the implemetation, we construct the **disparity matrix** that assign scores between two storms from consecutive frames.

Exploring the idea:
- Use hungarian matching to optimize the cost of matching storms from consecutive frames. The disadvantage of this is hard assigning and 
- Use disparity values to estimate the splitting, merging of storms.

- Source code for reference: folders `src/tracking/`

## Stage 3: Nowcasting
Use history of matching, build a nowcasting class that manage the movement history of storms. This class is also customized to make a prediction on the next position of the storm.

### Evaluation metrics
1. **Probability of Detection:** area of overlapping between predicted and grouth truth over total area of grouth truth. High POD means model detect region of convective cells well.
2. **False Alarm Ratio:** total area that system predict high DBZ but grouth truth have low DBZ value.
3. **Critical Success Index (CSI):** computed as the ratio of correctly predicted storms to the total area of predicted and true storms minus the area of correctly predicted storms.

*This need to be further considered.*


# C. Tracking analyzing
Because the movement vector seems to not perform better than **Benchmark** (assume that convective cells not change), hence for evaluating among difference tracking and identifying can be analyzed descriptively by:
- Change in the number of storms identified in consecutive scans.
- Movement history of matched cells through consecutive scans.
- Distribution of lifespan of storms.