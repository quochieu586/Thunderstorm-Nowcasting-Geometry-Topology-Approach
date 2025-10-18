# About this project

# 1. Project architecture
```
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

# 2. Implementation

This section gives a brief explanation about the `identification - tracking - nowcasting` framework that we will follow for implementing different methods.

## 2.1 Identification

In this step, the color images or dBz images are served as the input, and the results are a list of storms map by time. Each storm map contains the information about time of the current frame, and a list of storms.

Different identifications are performed, including:

- `SimpleIdentifier`: *TITAN_1993*

- `MorphIdentifier`: *ETITAN_2007*

- `MultiThreshIdentifier`: *ImprovedSCIT_2010*

- `HypoThreshIdentifier`: *AINT_2025*

They may have different hyperparameters, but sharing these common: `THRESHOLD` and `AREA_FILTER` (storm with area under this threshold will be canceled).

## 2.1.* Storm analysis

After identifying, storm and its attributes are stored in a class. The base class is $\textcolor{red}{\texttt{StormObject}}$ which contains 2 common attributes: `id`, `contour`. Different methods may have different way to represent the storm, and must inherit the base class. For example, some of them:

- $\textcolor{red}{\texttt{CentroidStorm}}$: stores an additional attribute `centroid` of the storm.

- $\textcolor{red}{\texttt{ShapeStorm}}$: stores an additional attribute `shape_vectors` which keeps tracks of particles and its shape vector of the storm.

- ...

The class $\textcolor{red}{\texttt{StormsMap}}$ is a storage for all storms at a specific time. It includes 2 attributes: `time_frame`, `storms`.

## 2.2 Tracking

3 base classes are implemented:
1. $\textcolor{red}{\texttt{Matcher}}$: receive 2 list of storms in the consecutive frames and *prior knowledge* (optional, based on the method), and aim to return a temporary assignment which may be adjusted later. 2 methods:
    - `_construct_disparity_matrix`: receive 2 storm lists and return its cost matrix. Note that the cost matrix may be modified (for example: using velocity constraint) before matching using Hungarian.
    - `_hungarian_matching`: return an optimal assigment.

2. $\textcolor{red}{\texttt{TrackingHistory}}$: keep track of storms evolution over time. 2 methods:
    - `forecast`: forecast the next position of current storm track.
    - `update`: receive the mapping of current storms and update the track.

3. $\textcolor{red}{\texttt{Tracker}}$: a common class that keep everything about the tracking step. 2 methods:
    - `fit`: receive the list of storms over time, match them step-by-step and record the tracking history.
    - `predict`: predict the next position of the current track.

Different methods may have different ways to implement these classes.

## 2.3 Nowcasting

`forecasting` api has been implemented in `Tracking` , since most of algorithms require to use the forecast together with the tracking step. Hence, nothing to be implemented more in this step.

