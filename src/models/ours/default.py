# storm tracking parameters
DEFAULT_MAX_VELOCITY = 100
DEFAULT_WEIGHTS = (0.5, 0.5)
DEFAULT_COARSE_MATCHING_THRESHOLD = 0.45
DEFAULT_FINE_MATCHING_THRESHOLD = 0.5

# storm identification parameters
DEFAULT_DBZ_THRESHOLD = 30
DEFAULT_DISTANCE_DBZ = 5
DEFAULT_FILTER_AREA = 20            # storm with area under this threshold => cancel
DEFAULT_FILTER_CENTER = 10

# shape vector construction parameters
DEFAULT_RADII = [30, 60, 90, 120]
DEFAULT_NUM_SECTORS = 8
DEFAULT_DENSITY = 0.05