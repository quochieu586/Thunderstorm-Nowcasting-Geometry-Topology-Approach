from datetime import datetime
from pathlib import Path

import nexradaws
import numpy as np
import pyart
import pytz





def main():
    """Download and process NEXRAD radar data."""
    # Initialize NEXRAD connection
    conn = nexradaws.NexradAwsInterface()

    # Configuration for data download
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    events = [
        {
            "radar": "KDVN",
            "label": "June 16, 2024 - Davenport, IA",
            "start_utc": datetime(2024, 6, 16, 16, 0, tzinfo=pytz.UTC),
            "end_utc": datetime(2024, 6, 16, 22, 0, tzinfo=pytz.UTC),
        },
        {
            "radar": "KGRR",
            "label": "May 7, 2024 - Grand Rapids, MI",
            "start_utc": datetime(2024, 5, 7, 22, 0, tzinfo=pytz.UTC),
            "end_utc": datetime(2024, 5, 8, 1, 0, tzinfo=pytz.UTC),
        },
        {
            "radar": "KARX",
            "label": "Aug 11, 2023 - La Crosse, WI",
            "start_utc": datetime(2023, 8, 11, 20, 0, tzinfo=pytz.UTC),
            "end_utc": datetime(2023, 8, 12, 5, 0, tzinfo=pytz.UTC),
        },
    ]

    # Download NEXRAD data for each event
    for event in events:
        radar = event["radar"]
        start_utc = event["start_utc"]
        end_utc = event["end_utc"]

        print(f"\n=== Processing {event['label']} | Radar: {radar} ===")
        print(f"UTC time: {start_utc} -> {end_utc}")

        # Get available scans in the specified time range
        scans = conn.get_avail_scans_in_range(start_utc, end_utc, radar)
        scans = [s for s in scans if "_MDM" not in s.filename]
        print(f"Found {len(scans)} total scans")

        # Create output directory and download data
        radar_dir = out_dir / radar / start_utc.strftime("%Y%m%d")
        radar_dir.mkdir(parents=True, exist_ok=True)

        conn.download(scans, radar_dir)
        print(f"Downloaded data to: {radar_dir.resolve()}")

    print("\nAll radar data downloaded.")
    print(f"Files saved in: {out_dir.resolve()}")


def process_radar_data():
    """Convert downloaded NEXRAD data to numpy grids."""
    # Configuration for radar data processing
    input_root = Path("./data")
    output_root = Path("./data/numpy_grid")
    output_root.mkdir(parents=True, exist_ok=True)

    grid_shape = (20, 901, 901)
    grid_limits = (
        (0, 10000),
        (-450000, 450000),
        (-450000, 450000),
    )

    radar_files = sorted(input_root.glob("*/*/*_V06"))

    print(f"Found {len(radar_files)} radar files")

    # Process and convert radar data to numpy grids
    for radar_file in radar_files:
        try:
            print(f"Processing {radar_file.name} ...")

            # Read NEXRAD archive file
            radar = pyart.io.read_nexrad_archive(str(radar_file))

            # Create grid from radar data
            grid = pyart.map.grid_from_radars(
                (radar,),
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                fields=["reflectivity"],
            )

            # Extract reflectivity data and compute 2D composite (max across heights)
            refl_data_3d = grid.fields["reflectivity"]["data"]
            composite_refl_2d = np.max(refl_data_3d, axis=0)

            # Handle fill values
            fill_value = grid.fields["reflectivity"].get("_FillValue", None)
            if fill_value is not None:
                composite_refl_2d = np.ma.masked_where(
                    composite_refl_2d == fill_value, composite_refl_2d
                )

            composite_refl_2d = composite_refl_2d.filled(np.nan)

            # Save processed data
            radar_name = radar_file.parent.parent.name  # KDVN / KGRR / KARX
            save_dir = output_root / radar_name
            save_dir.mkdir(parents=True, exist_ok=True)

            out_path = save_dir / f"{radar_file.name}.npy"
            np.save(out_path, composite_refl_2d)

            print(f"Saved -> {out_path}")

        except Exception as e:
            print(f"Failed on {radar_file.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
    process_radar_data()


