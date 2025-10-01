# About this project

# Project architecture
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