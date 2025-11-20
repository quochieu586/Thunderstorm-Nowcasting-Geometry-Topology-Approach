import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.cores.contours import StormObject, StormsMap
def make_storm_animation(storms_map_time_lst):
    """
    Create and return a FuncAnimation showing reflectivity (left)
    and reflectivity + storm overlays (right).
    """
    # === Setup figure with two subplots ===
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7))
    for ax in (ax_left, ax_right):
        ax.set_aspect("equal")

    # Initialize first frame
    first_map = storms_map_time_lst[0]

    if not hasattr(first_map, 'dbz_map'):
        raise AttributeError("This StormsMap instance dbz_map attribute is not found. Perhaps this storms tracking pipeline doesn't use a dbz_map?")
    
    # Left: raw field only
    im_left = ax_left.imshow(
        first_map.dbz_map,
        cmap="HomeyerRainbow",
        origin="upper",
        vmin=0,
        vmax=75,
    )
    ax_left.set_title("Raw Reflectivity")
    ax_left.set_xlabel("X (pixels)")
    ax_left.set_ylabel("Y (pixels)")


    # Right: reflectivity + storms
    im_right = ax_right.imshow(
        first_map.dbz_map,
        cmap="HomeyerRainbow",
        origin="upper",
        vmin=0,
        vmax=75,
    )
    ax_right.set_title(
        f"Frame 0: {first_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    ax_right.set_xlabel("X (pixels)")
    ax_right.set_ylabel("Y (pixels)")

    # === Update function ===
    def update(frame_idx):
        storms_map = storms_map_time_lst[frame_idx]

        # Left panel update
        im_left.set_data(storms_map.dbz_map)

        # Right panel update (clear + redraw)
        ax_right.clear()
        ax_right.imshow(
            storms_map.dbz_map,
            cmap="HomeyerRainbow",
            origin="upper",
            vmin=0,
            vmax=75,
        )

        # Overlay storm objects
        for storm in storms_map.storms:
            storm: StormObject
            storm.plot_on(ax_right)

        ax_right.set_title(
            f"Frame {frame_idx}: {storms_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        ax_right.set_xlabel("X (pixels)")
        ax_right.set_ylabel("Y (pixels)")
        ax_right.set_aspect("equal")

        return []

    # === Create animation ===
    anim = FuncAnimation(
        fig,
        update,
        frames=len(storms_map_time_lst),
        interval=500,
        blit=False,
        repeat=True,
    )
    plt.close(fig)
    return anim
