# Thunderstorm Nowcasting Visualization App ⛈️

A Streamlit-based web application for visualizing and analyzing thunderstorm data across multiple consecutive radar scans.

## Features

- **Multi-location Support**: Analyze radar data from various geographical regions
- **Multiple Storm Identification Methods**: Compare different algorithms (Simple Contour, Hypothesis, Morphology, Cluster)
- **Interactive Navigation**: Step through consecutive scans with ease
- **Real-time Processing**: Adjust parameters and see results instantly
- **Storm Analytics**: View detailed statistics about detected storms
- **Configurable Visualization**: Toggle contours, centroids, and labels

## Quick Start

### 1. Install Dependencies
```bash
# Install main project requirements
pip install -r ../requirements.txt

# Install app-specific requirements  
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Option 1: Use the launcher script
python run_app.py

# Option 2: Run directly with streamlit
streamlit run main.py
```

### 3. Access the App
Open your browser and navigate to `http://localhost:8501`

## Application Structure

```
app/
├── main.py                 # Main application entry point
├── run_app.py             # App launcher script  
├── requirements.txt       # App-specific dependencies
├── README.md             # This file
├── components/           # Reusable UI components
│   ├── __init__.py
│   ├── sidebar.py        # Dataset and parameter controls
│   ├── image_visualization.py  # Image display and overlays
│   └── data_processor.py # Data loading and processing
├── config/              # Configuration modules
│   ├── __init__.py
│   └── app_config.py    # App configuration and settings
└── pages/               # Application pages
    ├── __init__.py
    └── home_page.py     # Main visualization page
```

## Usage Guide

### 1. Dataset Selection
- Choose from available geographical locations in the sidebar
- Each dataset contains multiple consecutive radar scans
- Datasets support both image (.png) and GRIB2 formats

### 2. Identification Method
Select from available storm identification algorithms:
- **Simple Contour**: Basic threshold-based detection
- **Hypothesis**: Advanced subcell processing with dilation  
- **Morphology**: Morphological operations for storm detection
- **Cluster**: Clustering-based storm identification

### 3. Processing Parameters
- **DBZ Threshold**: Minimum reflectivity value for storm detection (20-50 dBZ)
- **Minimum Area Filter**: Filter out small objects (5-50 pixels)

### 4. Navigation
- Use the slider or Previous/Next buttons to navigate through scans
- Enable "Auto-process" for automatic processing when changing scans
- View scan information including filename and position

### 5. Visualization Options
- **Side-by-side Comparison**: Original image vs. processed DBZ map
- **DBZ Map Only**: Focus on the processed reflectivity data
- **Original Image Only**: View raw radar imagery
- Toggle storm contours, centroids, and labels

## Data Requirements

The app expects data to be organized in the following structure:

```
data/
├── images/                    # Image-based datasets
│   ├── location1/
│   │   ├── scan001.png
│   │   ├── scan002.png
│   │   └── ...
│   └── location2/
│       └── ...
└── MRMS_MergedReflectivityComposite_*/  # GRIB2 datasets
    ├── file001.grib2
    ├── file002.grib2
    └── ...
```

## Performance Tips

- **Caching**: The app caches processed images and storm data for faster navigation
- **Clear Cache**: Use the "Clear Cache" button if you modify processing parameters
- **Auto-process**: Disable auto-processing for manual control over when scans are processed

## Troubleshooting

### Common Issues

1. **No datasets found**: Ensure your data is properly organized in the `data/` directory
2. **Import errors**: Make sure all dependencies are installed and the project structure is intact
3. **Processing errors**: Check that your data files are valid and readable
4. **Performance issues**: Try clearing the cache or reducing the number of cached items

### Error Messages

- **Dataset not found**: Selected dataset no longer exists or path changed
- **Scan index out of range**: Trying to access a scan that doesn't exist  
- **Processing failed**: Issue with storm identification algorithm or data format

## Development

### Adding New Components

1. Create new component files in `app/components/`
2. Import in `app/components/__init__.py`
3. Use in pages or main application

### Adding New Pages

1. Create new page files in `app/pages/`
2. Import in `app/pages/__init__.py` 
3. Integrate with main navigation

### Configuration

Modify `app/config/app_config.py` to:
- Add new processing parameters
- Configure default values
- Add new data sources
- Customize UI settings

## License

This application is part of the Thunderstorm Nowcasting Geometry Approach project.
