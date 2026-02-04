"""
Animation module for storm visualization over time.

Provides functions to create animated visualizations of storm detection and particle tracking.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import cv2
from typing import List, Callable, Optional, Tuple
from datetime import datetime
import matplotlib as mpl

class StormAnimator:
    """
    A class to create animations of storm detection and particle visualization over time.
    
    Attributes:
        storms_map_list (List): List of storms maps over time.
        img_shape (tuple): Shape of the image (height, width).
        figsize (tuple): Figure size for the animation.
        interval (int): Interval between frames in milliseconds.
    """
    
    def __init__(
        self,
        storms_map_list: List,
        img_shape: tuple = (901, 901),
        figsize: tuple = (12, 10),
        interval: int = 500,
        window: Optional[tuple] = None
    ):
        """
        Initialize the StormAnimator.
        
        Parameters:
        -----------
        storms_map_list : List
            List of DbzStormsMap objects over time frames.
        img_shape : tuple, default=(901, 901)
            Shape of the image (height, width).
        figsize : tuple, default=(12, 10)
            Figure size for matplotlib.
        interval : int, default=500
            Interval between frames in milliseconds.
        window : tuple, optional
            Cropping window as (y_start, y_end, x_start, x_end) or None for full image.
            Example: (200, 700, 200, 700) displays a 500x500 region.
        """
        self.storms_map_list = storms_map_list
        self.img_shape = img_shape
        self.figsize = figsize
        self.interval = interval
        self.window = window
        self.fig = None
        self.ax = None
        self.anim = None
    
    def _draw_frame(
        self,
        frame_idx: int,
        draw_contours: bool = True,
        draw_particles: bool = True,
        contour_color: tuple = (100, 100, 100),
        contour_thickness: int = 2,
        particle_radius: int = 2
    ) -> np.ndarray:
        """
        Draw a single frame with storms and particles.
        
        Parameters:
        -----------
        frame_idx : int
            Index of the time frame.
        draw_contours : bool, default=True
            Whether to draw storm contours.
        draw_particles : bool, default=True
            Whether to draw particles.
        contour_color : tuple, default=(100, 100, 100)
            RGB color for contours.
        contour_thickness : int, default=2
            Thickness of contour lines.
        particle_radius : int, default=2
            Radius of particle circles.
        
        Returns:
        --------
        canvas : np.ndarray
            The drawn canvas image (or cropped window if window parameter is set).
        """
        storms_map = self.storms_map_list[frame_idx]
        
        # Create blank canvas
        canvas = np.ones(self.img_shape, dtype=np.uint8) * 255
        
        # Import here to avoid circular imports
        from src.preprocessing import convert_polygons_to_contours
        
        # Draw storm contours
        if draw_contours:
            for storm in storms_map.storms:
                cv2.drawContours(
                    canvas,
                    convert_polygons_to_contours([storm.contour]),
                    -1,
                    contour_color,
                    thickness=contour_thickness
                )
        
        # Draw particles
        if draw_particles:
            canvas = self._draw_particles(storms_map, canvas, particle_radius)
        
        # Apply window cropping if specified
        if self.window is not None:
            y_start, y_end, x_start, x_end = self.window
            canvas = canvas[y_start:y_end, x_start:x_end]
        
        return canvas
    
    def _draw_particles(self, storms_map, canvas: np.ndarray, particle_radius: int = 2) -> np.ndarray:
        """
        Draw particles from all storms on the canvas.
        
        Parameters:
        -----------
        storms_map : DbzStormsMap
            The storms map containing storms with shape vectors.
        canvas : np.ndarray
            The canvas to draw particles on.
        particle_radius : int, default=2
            Radius of particle circles.
        
        Returns:
        --------
        canvas : np.ndarray
            Updated canvas with particles drawn.
        """
        for storm_idx, storm in enumerate(storms_map.storms):
            # Generate a color for this storm
            color = tuple(np.random.randint(0, 200, size=3).tolist())
            
            # Draw each shape vector (particle) from the storm
            for shape_vector in storm.shape_vectors:
                # Extract coordinates (y, x) order from ShapeVector
                y, x = int(shape_vector.coord[0]), int(shape_vector.coord[1])
                
                # Draw particle as a small circle
                cv2.circle(canvas, (x, y), radius=particle_radius, color=color, thickness=-1)
        
        return canvas
    
    def create_animation(
        self,
        draw_contours: bool = True,
        draw_particles: bool = True,
        contour_color: tuple = (100, 100, 100),
        contour_thickness: int = 2,
        particle_radius: int = 2,
        cmap: str = 'gray',
        title_template: Optional[str] = None
    ) -> animation.FuncAnimation:
        """
        Create an animation of storm detection and particle visualization.
        
        Parameters:
        -----------
        draw_contours : bool, default=True
            Whether to draw storm contours.
        draw_particles : bool, default=True
            Whether to draw particles.
        contour_color : tuple, default=(100, 100, 100)
            RGB color for contours.
        contour_thickness : int, default=2
            Thickness of contour lines.
        particle_radius : int, default=2
            Radius of particle circles in pixels.
        cmap : str, default='gray'
            Colormap for displaying the image.
        title_template : str, optional
            Template for the title. Use {frame_idx} and {time} placeholders.
            Default: 'Storm Detection - Time Frame {frame_idx}\n{time} | Storms: {storms}'
            Note: Window region info can be added via custom template.
        
        Returns:
        --------
        anim : animation.FuncAnimation
            The matplotlib animation object.
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        if title_template is None:
            title_template = 'Storm Detection - Time Frame {frame_idx}\n{time} | Storms: {storms}'
        
        def animate(frame_idx):
            """Animate function for each frame."""
            self.ax.clear()
            
            storms_map = self.storms_map_list[frame_idx]
            
            # Draw frame
            canvas = self._draw_frame(
                frame_idx,
                draw_contours=draw_contours,
                draw_particles=draw_particles,
                contour_color=contour_color,
                contour_thickness=contour_thickness,
                particle_radius=particle_radius
            )
            
            # Display the image
            self.ax.imshow(canvas, cmap=cmap)
            
            # Format title
            time_str = storms_map.time_frame.strftime("%Y-%m-%d %H:%M:%S")
            window_str = f" (Window: {self.window})" if self.window is not None else ""
            title = title_template.format(
                frame_idx=frame_idx,
                time=time_str,
                storms=len(storms_map.storms)
            ) + window_str
            
            self.ax.set_title(title, fontsize=12, fontweight='bold')
            self.ax.set_xlabel('X Coordinate' if self.window is None else 'X (windowed)')
            self.ax.set_ylabel('Y Coordinate' if self.window is None else 'Y (windowed)')
            
            return self.ax,
        
        num_frames = len(self.storms_map_list)
        print(f"Creating animation with {num_frames} frames...")
        
        self.anim = animation.FuncAnimation(
            self.fig,
            animate,
            frames=num_frames,
            interval=self.interval,
            repeat=True,
            blit=False
        )
        
        plt.close(self.fig)  # Close figure to prevent duplicate display
        
        return self.anim
    
    def to_html(self) -> HTML:
        """
        Convert animation to HTML for Jupyter notebook display.
        
        Returns:
        --------
        HTML : IPython.display.HTML
            HTML representation of the animation.
        """
        if self.anim is None:
            raise ValueError("Animation not created. Call create_animation() first.")
        return HTML(self.anim.to_jshtml())
    
    def save(self, filepath: str, writer: str = 'ffmpeg', fps: int = 2):
        """
        Save animation to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the animation file.
        writer : str, default='ffmpeg'
            Writer to use (e.g., 'ffmpeg', 'pillow').
        fps : int, default=2
            Frames per second for the output video.
        """
        if self.anim is None:
            raise ValueError("Animation not created. Call create_animation() first.")
        self.anim.save(filepath, writer=writer, fps=fps)
        print(f"Animation saved to {filepath}")

def animate_storms(
    storms_map_list: List,
    img_shape: tuple = (901, 901),
    figsize: tuple = (12, 10),
    interval: int = 500,
    draw_contours: bool = True,
    draw_particles: bool = True,
    contour_color: tuple = (100, 100, 100),
    contour_thickness: int = 2,
    particle_radius: int = 2,
    cmap: str = 'gray',
    title_template: Optional[str] = None,
    window: Optional[tuple] = None,
    return_animator: bool = False
) -> Optional[HTML]:
    """
    Create and display a storm animation.
    
    Parameters:
    -----------
    storms_map_list : List
        List of DbzStormsMap objects over time frames.
    img_shape : tuple, default=(901, 901)
        Shape of the image (height, width).
    figsize : tuple, default=(12, 10)
        Figure size for matplotlib.
    interval : int, default=500
        Interval between frames in milliseconds.
    draw_contours : bool, default=True
        Whether to draw storm contours.
    draw_particles : bool, default=True
        Whether to draw particles.
    contour_color : tuple, default=(100, 100, 100)
        RGB color for contours.
    contour_thickness : int, default=2
        Thickness of contour lines.
    particle_radius : int, default=2
        Radius of particle circles in pixels.
    cmap : str, default='gray'
        Colormap for displaying the image.
    title_template : str, optional
        Template for the title. Use {frame_idx} and {time} placeholders.
    window : tuple, optional
        Cropping window as (y_start, y_end, x_start, x_end) or None for full image.
        Example: (200, 700, 200, 700) displays a 500x500 region.
    return_animator : bool, default=False
        If True, return the StormAnimator object instead of HTML.
    
    Returns:
    --------
    result : HTML or StormAnimator
        If return_animator=False, returns HTML for display.
        If return_animator=True, returns StormAnimator object.
    
    Example:
    --------
    >>> from src.animation import animate_storms
    >>> # Full image animation
    >>> animate_storms(storms_map_list)
    
    >>> # Cropped region animation (e.g., 500x500 from (200,700,200,700))
    >>> animate_storms(storms_map_list, window=(200, 700, 200, 700))
    
    >>> # Advanced with animator object
    >>> animator = animate_storms(storms_map_list, return_animator=True)
    >>> animator.create_animation()
    """
    animator = StormAnimator(
        storms_map_list=storms_map_list,
        img_shape=img_shape,
        figsize=figsize,
        interval=interval,
        window=window
    )
    
    animator.create_animation(
        draw_contours=draw_contours,
        draw_particles=draw_particles,
        contour_color=contour_color,
        contour_thickness=contour_thickness,
        particle_radius=particle_radius,
        cmap=cmap,
        title_template=title_template
    )
    
    if return_animator:
        return animator
    
    return animator.to_html()

def animate_dbz_map(
    storms_map_list: List,
    figsize: tuple = (12, 10),
    interval: int = 500,
    cmap: str = 'viridis',
    title_template: Optional[str] = None,
    window: Optional[tuple] = None,
    return_animator: bool = False
) -> Optional[HTML]:
    """
    Create and display an animation of DBZ maps over time.
    
    Parameters:
    -----------
    storms_map_list : List
        List of DbzStormsMap objects over time frames.
    figsize : tuple, default=(12, 10)
        Figure size for matplotlib.
    interval : int, default=500
        Interval between frames in milliseconds.
    cmap : str, default='viridis'
        Colormap for displaying the DBZ map.
    title_template : str, optional
        Template for the title. Use {frame_idx} and {time} placeholders.
        Default: 'DBZ Map - Time Frame {frame_idx}\n{time}'
    window : tuple, optional
        Cropping window as (y_start, y_end, x_start, x_end) or None for full image.
        Example: (200, 700, 200, 700) displays a 500x500 region.
    return_animator : bool, default=False
        If True, return a tuple (fig, anim) for further customization.
    
    Returns:
    --------
    result : HTML or tuple
        If return_animator=False, returns HTML for display.
        If return_animator=True, returns tuple (fig, anim).
    
    Example:
    --------
    >>> from src.animation import animate_dbz_map
    >>> # Full image DBZ map animation
    >>> animate_dbz_map(storms_map_list)
    
    >>> # Cropped region (e.g., 500x500 from (200,700,200,700))
    >>> animate_dbz_map(storms_map_list, window=(200, 700, 200, 700))
    
    >>> # Advanced: customize colormap
    >>> animate_dbz_map(storms_map_list, cmap='plasma', interval=300, window=(100, 800, 100, 800))
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if title_template is None:
        title_template = 'DBZ Map - Time Frame {frame_idx}\n{time}'
    
    # Create colorbar once outside the animation function
    cbar = None
    
    def animate(frame_idx):
        """Animate function for each frame."""
        nonlocal cbar
        ax.clear()
        
        storms_map = storms_map_list[frame_idx]
        dbz_map = storms_map.dbz_map
        
        # Apply window cropping if specified
        if window is not None:
            y_start, y_end, x_start, x_end = window
            dbz_map = dbz_map[y_start:y_end, x_start:x_end]
        
        # Display DBZ map
        im = ax.imshow(dbz_map, cmap=cmap)
        
        # Format title
        time_str = storms_map.time_frame.strftime("%Y-%m-%d %H:%M:%S")
        window_str = f" (Window: {window})" if window is not None else ""
        title = title_template.format(
            frame_idx=frame_idx,
            time=time_str
        ) + window_str
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate' if window is None else 'X (windowed)')
        ax.set_ylabel('Y Coordinate' if window is None else 'Y (windowed)')
                
        return ax,
    
    num_frames = len(storms_map_list)
    print(f"Creating DBZ map animation with {num_frames} frames...")
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval,
        repeat=True,
        blit=False
    )
    
    if return_animator:
        return fig, anim
    
    plt.close(fig)  # Close figure to prevent duplicate display
    
    return HTML(anim.to_jshtml())

def animate_side_by_side(
    storms_map_list: List,
    img_shape: tuple = (901, 901),
    figsize: tuple = (18, 8),
    interval: int = 500,
    draw_contours: bool = True,
    draw_particles: bool = True,
    contour_color: tuple = (100, 100, 100),
    contour_thickness: int = 2,
    storms_cmap: str = 'gray',
    dbz_cmap: str = 'viridis',
    title_template: Optional[str] = None,
    window: Optional[tuple] = None,
    return_animator: bool = False
) -> Optional[HTML]:
    """
    Create and display a side-by-side animation of DBZ map and storm detection.
    
    Parameters:
    -----------
    storms_map_list : List
        List of DbzStormsMap objects over time frames.
    img_shape : tuple, default=(901, 901)
        Shape of the image (height, width).
    figsize : tuple, default=(18, 8)
        Figure size for matplotlib (wider for side-by-side).
    interval : int, default=500
        Interval between frames in milliseconds.
    draw_contours : bool, default=True
        Whether to draw storm contours.
    draw_particles : bool, default=True
        Whether to draw particles.
    contour_color : tuple, default=(100, 100, 100)
        RGB color for contours.
    contour_thickness : int, default=2
        Thickness of contour lines.
    storms_cmap : str, default='gray'
        Colormap for storm detection display.
    dbz_cmap : str, default='viridis'
        Colormap for DBZ map display.
    title_template : str, optional
        Template for the main title. Use {frame_idx} and {time} placeholders.
    window : tuple, optional
        Cropping window as (y_start, y_end, x_start, x_end) or None for full image.
        Applied to both DBZ map and storm detection for synchronized cropping.
        Example: (200, 700, 200, 700) displays a 500x500 region.
    return_animator : bool, default=False
        If True, return a tuple (fig, anim) for further customization.
    
    Returns:
    --------
    result : HTML or tuple
        If return_animator=False, returns HTML for display.
        If return_animator=True, returns tuple (fig, anim).
    
    Example:
    --------
    >>> from src.animation import animate_side_by_side
    >>> # Full image side-by-side animation
    >>> animate_side_by_side(storms_map_list)
    
    >>> # Cropped region (e.g., 500x500 from (200,700,200,700))
    >>> animate_side_by_side(storms_map_list, window=(200, 700, 200, 700))
    
    >>> # Advanced: customize colormaps with window cropping
    >>> animate_side_by_side(
    ...     storms_map_list,
    ...     dbz_cmap='plasma',
    ...     storms_cmap='gray',
    ...     interval=300,
    ...     window=(100, 800, 100, 800)
    ... )
    """
    fig, (ax_dbz, ax_storms) = plt.subplots(1, 2, figsize=figsize)
    
    if title_template is None:
        title_template = 'Time Frame {frame_idx} - {time}'
    
    # Create animator for storm detection
    animator = StormAnimator(
        storms_map_list=storms_map_list,
        img_shape=img_shape,
        figsize=figsize,
        interval=interval,
        window=window
    )
    
    # Colorbar for DBZ map
    cbar = None
    
    def animate(frame_idx):
        """Animate function for each frame."""
        nonlocal cbar
        
        # Clear axes
        ax_dbz.clear()
        ax_storms.clear()
        
        storms_map = storms_map_list[frame_idx]
        
        # Left: DBZ Map
        dbz_map = storms_map.dbz_map
        if window is not None:
            y_start, y_end, x_start, x_end = window
            dbz_map = dbz_map[y_start:y_end, x_start:x_end]
        
        im_dbz = ax_dbz.imshow(dbz_map, cmap=dbz_cmap)
        ax_dbz.set_title('DBZ Map', fontsize=11, fontweight='bold')
        ax_dbz.set_xlabel('X Coordinate' if window is None else 'X (windowed)')
        ax_dbz.set_ylabel('Y Coordinate' if window is None else 'Y (windowed)')
        
        # Right: Storm Detection
        canvas = animator._draw_frame(
            frame_idx,
            draw_contours=draw_contours,
            draw_particles=draw_particles,
            contour_color=contour_color,
            contour_thickness=contour_thickness
        )
        
        ax_storms.imshow(canvas, cmap=storms_cmap)
        ax_storms.set_title('Storm Detection', fontsize=11, fontweight='bold')
        ax_storms.set_xlabel('X Coordinate' if window is None else 'X (windowed)')
        ax_storms.set_ylabel('Y Coordinate' if window is None else 'Y (windowed)')
        
        # Main title
        time_str = storms_map.time_frame.strftime("%Y-%m-%d %H:%M:%S")
        window_str = f" (Window: {window})" if window is not None else ""
        title = title_template.format(
            frame_idx=frame_idx,
            time=time_str
        ) + window_str
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return ax_dbz, ax_storms
    
    num_frames = len(storms_map_list)
    print(f"Creating side-by-side animation with {num_frames} frames...")
    print("  Left: DBZ Map | Right: Storm Detection")
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval,
        repeat=True,
        blit=False
    )
    
    if return_animator:
        return fig, anim
    
    plt.close(fig)  # Close figure to prevent duplicate display
    
    return HTML(anim.to_jshtml())

def animate_particle_matching(
    storms_map_list: List,
    particle_assignments_list: List[np.ndarray],
    img_shape: tuple = (901, 901),
    figsize: tuple = (16, 8),
    interval: int = 500,
    cmap: str = "viridis",   # default colormap
    vmin: float = 0.0,       # 
    vmax: float = 80.0,      # 
    particle_radius: int = 3,
    line_thickness: int = 1,
    line_color: tuple = (0, 255, 0),          # BGR green
    particle_color_prev: tuple = (0, 0, 255), # BGR red
    particle_color_curr: tuple = (255, 0, 0), # BGR blue
    save_path: Optional[str] = None,
    fps: int = 2
) -> Optional[HTML]:
    """
    Animate particle matching between consecutive frames on ONE combined canvas:
    - Old frame on the LEFT (dbz_map background)
    - New frame on the RIGHT (dbz_map background)
    - Lines connect matched particles across halves
    - dbz_map is colorized using a fixed colormap normalization [vmin, vmax]
    """

    if len(particle_assignments_list) != len(storms_map_list) - 1:
        raise ValueError(
            f"particle_assignments_list should have {len(storms_map_list) - 1} elements "
            f"but got {len(particle_assignments_list)}"
        )

    H, W = img_shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")

    def get_particle_coords(storms_map):
        """particle_index -> (x, y), flattening storms then shape_vectors"""
        coords_by_idx = {}
        idx = 0
        for storm in storms_map.storms:
            for sv in storm.shape_vectors:
                coords_by_idx[idx] = sv.coord
                idx += 1
        return coords_by_idx

    def dbz_to_bgr(dbz_map: np.ndarray, cmap_name: str, vmin_: float, vmax_: float) -> np.ndarray:
        """
        Convert 2D dbz_map to a colored BGR uint8 image using matplotlib colormap,
        with fixed normalization [vmin_, vmax_].
        """
        dbz = dbz_map.astype(np.float32)

        # clip then normalize using fixed range
        dbz = np.clip(dbz, vmin_, vmax_)
        denom = max(vmax_ - vmin_, 1e-6)
        norm = (dbz - vmin_) / denom  # [0..1]

        cmap_fn = mpl.colormaps.get_cmap(cmap_name)
        rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)  # RGB
        bgr = rgb[..., ::-1]  # RGB -> BGR
        return bgr

    def animate(frame_idx):
        ax.clear()
        ax.axis("off")

        prev_map = storms_map_list[frame_idx]
        curr_map = storms_map_list[frame_idx + 1]
        assignments = particle_assignments_list[frame_idx]

        prev_coords = get_particle_coords(prev_map)
        curr_coords = get_particle_coords(curr_map)

        # --- DBZ background (left = prev, right = curr) ---
        prev_bg = dbz_to_bgr(prev_map.dbz_map, cmap, vmin, vmax)
        curr_bg = dbz_to_bgr(curr_map.dbz_map, cmap, vmin, vmax)

        # resize if needed
        if prev_bg.shape[:2] != (H, W):
            prev_bg = cv2.resize(prev_bg, (W, H))
        if curr_bg.shape[:2] != (H, W):
            curr_bg = cv2.resize(curr_bg, (W, H))

        canvas = np.zeros((H, 2 * W, 3), dtype=np.uint8)
        canvas[:, :W] = prev_bg
        canvas[:, W:2 * W] = curr_bg

        # --- Draw particles ---
        for (x, y) in prev_coords.values():
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), particle_radius, particle_color_prev, -1)

        for (x, y) in curr_coords.values():
            x, y = int(x), int(y)
            x_shift = x + W
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x_shift, y), particle_radius, particle_color_curr, -1)

        # --- Draw matching lines ---
        for prev_idx, curr_idx in assignments:
            if prev_idx not in prev_coords or curr_idx not in curr_coords:
                continue

            x_prev, y_prev = prev_coords[prev_idx]
            x_curr, y_curr = curr_coords[curr_idx]

            x_prev, y_prev = int(x_prev), int(y_prev)
            x_curr, y_curr = int(x_curr) + W, int(y_curr)

            if (0 <= x_prev < W and 0 <= y_prev < H and
                W <= x_curr < 2 * W and 0 <= y_curr < H):
                cv2.line(canvas, (x_prev, y_prev), (x_curr, y_curr), line_color, line_thickness)

        # divider line
        cv2.line(canvas, (W, 0), (W, H - 1), (0, 0, 0), 2)

        # show
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        match_rate = len(assignments) / max(len(prev_coords), len(curr_coords), 1) * 100
        ax.set_title(
            f"Frame {frame_idx} → {frame_idx + 1} | "
            f"Matched: {len(assignments)} | Match Rate: {match_rate:.1f}%\n"
            f"Left: {prev_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')}   |   "
            f"Right: {curr_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"cmap={cmap}, vmin={vmin:g}, vmax={vmax:g}",
            fontsize=11,
            fontweight="bold"
        )

        return ax

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(particle_assignments_list),
        interval=interval,
        repeat=True,
        blit=False
    )

    # save if needed
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        anim.save(save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        return None

    plt.close(fig)
    return HTML(anim.to_jshtml())

def animate_storm_matching(
    storms_map_list: List,
    matched_pairs_list: List[List],   # list of list[MatchedStormPair] for each (t->t+1)
    figsize: tuple = (16, 8),
    interval: int = 500,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 80.0,
    draw_contours: bool = True,
    contour_color: tuple = (180, 180, 180),   # BGR light gray
    contour_thickness: int = 2,
    centroid_radius: int = 4,
    centroid_color_prev: tuple = (0, 0, 255), # BGR red
    centroid_color_curr: tuple = (255, 0, 0), # BGR blue
    line_color: tuple = (0, 255, 0),          # BGR green
    line_thickness: int = 2,
    show_indices: bool = True,
    font_scale: float = 0.5,
    text_thickness: int = 1,
    window: Optional[tuple] = None,           # (y0,y1,x0,x1)
    save_path: Optional[str] = None,
    fps: int = 2,
) -> Optional[HTML]:
    """
    Animate storm-to-storm matching (storm-level).
    - Left: prev frame
    - Right: curr frame
    - Lines connect matched storm centroids (computed from contour moments)

    IMPORTANT:
    - Uses dbz_map native resolution (NO resizing) to avoid offset issues.
    - Centroids are computed from contour geometry in the same coordinate system.
    - Split/merge automatically appear as multiple lines.
    """

    if len(matched_pairs_list) != len(storms_map_list) - 1:
        raise ValueError(
            f"matched_pairs_list must have {len(storms_map_list)-1} elements, "
            f"but got {len(matched_pairs_list)}"
        )

    # Import here to avoid circular imports
    from src.preprocessing import convert_polygons_to_contours

    # Use native DBZ resolution
    H0, W0 = storms_map_list[0].dbz_map.shape

    def dbz_to_bgr(dbz_map: np.ndarray, cmap_name: str, vmin_: float, vmax_: float) -> np.ndarray:
        """
        Convert 2D dbz_map -> BGR uint8 image with fixed normalization [vmin, vmax].
        IMPORTANT: Return a contiguous BGR array (OpenCV requires this).
        """
        dbz = dbz_map.astype(np.float32)
        dbz = np.clip(dbz, vmin_, vmax_)
        denom = max(vmax_ - vmin_, 1e-6)
        norm = (dbz - vmin_) / denom  # [0..1]

        cmap_fn = mpl.colormaps.get_cmap(cmap_name)
        rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)  # RGB uint8

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def get_centroid_from_contour(storm) -> Optional[tuple[float, float]]:
        """Centroid (x,y) from contour using cv2.moments (DBZ pixel coords)."""
        if not hasattr(storm, "contour") or storm.contour is None:
            return None

        cnts = convert_polygons_to_contours([storm.contour])
        if cnts is None or len(cnts) == 0:
            return None

        cnt = np.asarray(cnts[0], dtype=np.int32)
        M = cv2.moments(cnt)

        if abs(M["m00"]) < 1e-6:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (float(cx), float(cy))

    def crop_point(x: float, y: float) -> tuple[float, float]:
        """Shift point if window is applied."""
        if window is None:
            return x, y
        y0, y1, x0, x1 = window
        return x - x0, y - y0

    def render_storm_frame(storms_map) -> np.ndarray:
        """
        Render one storms_map to BGR canvas:
          - DBZ background
          - contours
          - indices (optional)
        """
        dbz_map = storms_map.dbz_map
        if dbz_map.shape != (H0, W0):
            raise ValueError(f"Inconsistent dbz_map size: got {dbz_map.shape}, expected {(H0, W0)}")

        canvas = dbz_to_bgr(dbz_map, cmap, vmin, vmax)
        canvas = np.ascontiguousarray(canvas)

        # Draw contours
        if draw_contours:
            for storm in storms_map.storms:
                if not hasattr(storm, "contour") or storm.contour is None:
                    continue

                contours = convert_polygons_to_contours([storm.contour])
                if contours is None or len(contours) == 0:
                    continue

                # ensure correct dtype
                contours = [np.asarray(c, dtype=np.int32) for c in contours]

                cv2.drawContours(
                    canvas,
                    contours,
                    -1,
                    contour_color,
                    thickness=contour_thickness
                )

        # Draw indices (optional)
        if show_indices:
            for idx, storm in enumerate(storms_map.storms):
                ctr = get_centroid_from_contour(storm)
                if ctr is None:
                    continue
                x, y = int(ctr[0]), int(ctr[1])
                if 0 <= x < W0 and 0 <= y < H0:
                    cv2.putText(
                        canvas,
                        str(idx),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        text_thickness,
                        cv2.LINE_AA
                    )

        # Crop
        if window is not None:
            y0, y1, x0, x1 = window
            canvas = canvas[y0:y1, x0:x1].copy()  #

        return canvas

    def get_centroids_dict(storms_map) -> dict[int, tuple[float, float]]:
        """storm_idx -> (x,y) centroid in ORIGINAL (non-windowed) coords."""
        d = {}
        for idx, storm in enumerate(storms_map.storms):
            ctr = get_centroid_from_contour(storm)
            if ctr is not None:
                d[idx] = ctr
        return d

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis("off")

    def animate(i):
        ax.clear()
        ax.axis("off")

        prev_map = storms_map_list[i]
        curr_map = storms_map_list[i + 1]
        pairs = matched_pairs_list[i]

        prev_img = render_storm_frame(prev_map)
        curr_img = render_storm_frame(curr_map)

        h_vis, w_vis = prev_img.shape[:2]
        canvas = np.zeros((h_vis, 2 * w_vis, 3), dtype=np.uint8)
        canvas[:, :w_vis] = prev_img
        canvas[:, w_vis:2 * w_vis] = curr_img

        prev_centroids = get_centroids_dict(prev_map)
        curr_centroids = get_centroids_dict(curr_map)

        # Draw centroids
        for _, (x, y) in prev_centroids.items():
            x2, y2 = crop_point(x, y)
            x2, y2 = int(x2), int(y2)
            if 0 <= x2 < w_vis and 0 <= y2 < h_vis:
                cv2.circle(canvas, (x2, y2), centroid_radius, centroid_color_prev, -1)

        for _, (x, y) in curr_centroids.items():
            x2, y2 = crop_point(x, y)
            x2, y2 = int(x2) + w_vis, int(y2)
            if w_vis <= x2 < 2 * w_vis and 0 <= y2 < h_vis:
                cv2.circle(canvas, (x2, y2), centroid_radius, centroid_color_curr, -1)

        # Draw lines + split/merge counts
        prev_to_curr = {}
        curr_to_prev = {}

        for pair in pairs:
            p = pair.prev_storm_order
            c = pair.curr_storm_order

            prev_to_curr.setdefault(p, []).append(c)
            curr_to_prev.setdefault(c, []).append(p)

            if p not in prev_centroids or c not in curr_centroids:
                continue

            x_prev, y_prev = prev_centroids[p]
            x_curr, y_curr = curr_centroids[c]

            x_prev, y_prev = crop_point(x_prev, y_prev)
            x_curr, y_curr = crop_point(x_curr, y_curr)

            pt1 = (int(x_prev), int(y_prev))
            pt2 = (int(x_curr) + w_vis, int(y_curr))

            if (0 <= pt1[0] < w_vis and 0 <= pt1[1] < h_vis and
                w_vis <= pt2[0] < 2 * w_vis and 0 <= pt2[1] < h_vis):
                cv2.line(canvas, pt1, pt2, line_color, line_thickness)

        cv2.line(canvas, (w_vis, 0), (w_vis, h_vis - 1), (0, 0, 0), 2)

        num_splits = sum(1 for _, v in prev_to_curr.items() if len(v) > 1)
        num_merges = sum(1 for _, v in curr_to_prev.items() if len(v) > 1)

        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Storm Matching: Frame {i} → {i+1} | Pairs: {len(pairs)} | "
            f"Splits: {num_splits} | Merges: {num_merges}\n"
            f"Left: {prev_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')}   |   "
            f"Right: {curr_map.time_frame.strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=11,
            fontweight="bold"
        )

        return ax

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(matched_pairs_list),
        interval=interval,
        repeat=True,
        blit=False
    )

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        anim.save(save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        return None

    plt.close(fig)
    return HTML(anim.to_jshtml())

def animate_storm_custom(
    storms_map_list: List,
    img_shape: tuple = (901, 901),
    figsize: tuple = (18, 8),
    interval: int = 500,
    draw_contours: bool = True,
    draw_particles: bool = True,
    contour_color: tuple = (100, 100, 100),
    contour_thickness: int = 2,
    storms_cmap: str = "gray",
    vmin: float = 0.0,            
    vmax: float = 80.0,           
    dbz_cmap: str = "viridis",
    title_template: Optional[str] = None,
    window: Optional[tuple] = None,
    save_path: Optional[str] = None,
    fps: int = 2,
    n_axes: int = 3,
    custom_axis_drawers: Optional[List[Callable[[plt.Axes, int], None]]] = None,
    custom_titles: Optional[List[str]] = None,
    ncols: int = 2,  #  NEW: control grid width
):
    # -----------------------------
    # Helper: grid layout + stretch last axis
    # -----------------------------
    def make_axes_with_stretched_last(n_axes, figsize=(18, 8), ncols=2):
        import math

        nrows = math.ceil(n_axes / ncols)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows, ncols)

        axes = []
        for i in range(n_axes):
            r = i // ncols
            c = i % ncols

            is_last_axis = (i == n_axes - 1)
            slots_left_in_row = (ncols - 1) - c

            # ✅ stretch only the LAST axis if row incomplete
            if is_last_axis and slots_left_in_row > 0:
                ax = fig.add_subplot(gs[r, c:])
            else:
                ax = fig.add_subplot(gs[r, c])

            axes.append(ax)

        return fig, axes

    # -----------------------------
    # Setup
    # -----------------------------
    n_axes = max(2, n_axes)

    fig, axes = make_axes_with_stretched_last(n_axes, figsize=figsize, ncols=ncols)

    # nicer spacing (do NOT do tight_layout inside animate loop)
    fig.subplots_adjust(wspace=0.25, hspace=0.35)

    if title_template is None:
        title_template = "Time Frame {frame_idx} - {time}"

    # Track colorbar
    cbar = None

    # Your storm animator (assumed to exist)
    animator = StormAnimator(
        storms_map_list=storms_map_list,
        img_shape=img_shape,
        figsize=figsize,
        interval=interval,
        window=window,
    )

    # -----------------------------
    # Frame function
    # -----------------------------
    def animate(frame_idx):
        nonlocal cbar
        # Remove extra axes but preserve colorbar axis
        cbar_ax = cbar.ax if cbar is not None else None
        for extra_ax in list(fig.axes):
            if extra_ax is cbar_ax:
                continue
            if extra_ax not in axes:
                fig.delaxes(extra_ax)


        # clear all axes
        for ax in axes:
            ax.clear()

        storms_map = storms_map_list[frame_idx]

        ax_dbz = axes[0]
        ax_storms = axes[1]

        # ---- DBZ Map (left/top)
        dbz_map = storms_map.dbz_map
        if window is not None:
            y_start, y_end, x_start, x_end = window
            dbz_map = dbz_map[y_start:y_end, x_start:x_end]

        im_dbz = ax_dbz.imshow(dbz_map, cmap=dbz_cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax_dbz.set_title("DBZ Map", fontsize=11, fontweight="bold")
        ax_dbz.set_xlabel("X Coordinate" if window is None else "X (windowed)")
        ax_dbz.set_ylabel("Y Coordinate" if window is None else "Y (windowed)")

        # colorbar (created once, updated thereafter)
        if cbar is None:
            cbar = fig.colorbar(im_dbz, ax=ax_dbz, fraction=0.046, pad=0.02)
            cbar.set_label("dBZ", fontsize=10)
        else:
            cbar.update_normal(im_dbz)

        # ---- Storm Detection (right/top)
        storm_canvas = animator._draw_frame(
            frame_idx,
            draw_contours=draw_contours,
            draw_particles=draw_particles,
            contour_color=contour_color,
            contour_thickness=contour_thickness,
        )

        ax_storms.imshow(storm_canvas, cmap=storms_cmap, aspect="auto")
        ax_storms.set_title("Storm Detection", fontsize=11, fontweight="bold")
        ax_storms.set_xlabel("X Coordinate" if window is None else "X (windowed)")
        ax_storms.set_ylabel("Y Coordinate" if window is None else "Y (windowed)")

        # ---- Custom axes (below / extra slots)
        for ax_idx in range(2, n_axes):
            ax_custom = axes[ax_idx]
            custom_i = ax_idx - 2

            if custom_axis_drawers is not None and custom_i < len(custom_axis_drawers):
                drawer_fn = custom_axis_drawers[custom_i]
                drawer_fn(ax_custom, frame_idx)  #  draw on this axis

                if custom_titles is not None and custom_i < len(custom_titles):
                    ax_custom.set_title(custom_titles[custom_i], fontsize=11, fontweight="bold")
                else:
                    ax_custom.set_title(f"Custom Plot {ax_idx+1}", fontsize=11, fontweight="bold")
            else:
                ax_custom.set_title(f"Custom Axis {ax_idx+1}", fontsize=11, fontweight="bold")
                ax_custom.text(
                    0.5, 0.5, "Add your custom plots here",
                    ha="center", va="center",
                    transform=ax_custom.transAxes,
                    fontsize=12, color="gray"
                )
                ax_custom.set_xticks([])
                ax_custom.set_yticks([])


        # ---- Main title
        time_str = storms_map.time_frame.strftime("%Y-%m-%d %H:%M:%S")
        window_str = f" (Window: {window})" if window is not None else ""
        title = title_template.format(frame_idx=frame_idx, time=time_str) + window_str
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

        return axes

    # -----------------------------
    # Build animation
    # -----------------------------
    num_frames = len(storms_map_list)
    print(f"Creating animation with {num_frames} frames...")
    print("  DBZ Map | Storm Detection | Custom Panels")

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval,
        repeat=True,
        blit=False,
    )

    # -----------------------------
    # Save or return HTML
    # -----------------------------
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        anim.save(save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        return None

    plt.close(fig)
    return HTML(anim.to_jshtml())

def animate_storm_tracking_history(
    storms_map_list: List,
    figsize: tuple = (14, 10),
    interval: int = 500,
    cmap: str = 'viridis',
    vmin: float = 0.0,
    vmax: float = 80.0,
    current_storm_color: tuple = (255, 0, 0),  # BGR blue
    past_storm_color: tuple = (0, 0, 255),     # BGR red
    trajectory_color: tuple = (0, 255, 0),     # BGR green
    contour_thickness: int = 2,
    trajectory_thickness: int = 1,
    centroid_radius: int = 4,
    show_indices: bool = True,
    font_scale: float = 0.6,
    text_thickness: int = 2,
    max_history_frames: int = 10,
    window: Optional[tuple] = None,
    save_path: Optional[str] = None,
    fps: int = 2,
) -> Optional[HTML]:
    """
    Animate storm tracking history showing:
    - Current storms in blue
    - Past storm positions in red
    - Trajectory lines in green with storm indices
    
    Parameters:
    -----------
    storms_map_list : List
        List of DbzStormsMap objects over time frames.
    figsize : tuple, default=(14, 10)
        Figure size for matplotlib.
    interval : int, default=500
        Interval between frames in milliseconds.
    cmap : str, default='viridis'
        Colormap for DBZ background.
    vmin : float, default=0.0
        Minimum value for colormap normalization.
    vmax : float, default=80.0
        Maximum value for colormap normalization.
    current_storm_color : tuple, default=(255, 0, 0)
        BGR color for current storm contours (blue).
    past_storm_color : tuple, default=(0, 0, 255)
        BGR color for past storm contours (red).
    trajectory_color : tuple, default=(0, 255, 0)
        BGR color for trajectory lines (green).
    contour_thickness : int, default=2
        Thickness of contour lines.
    trajectory_thickness : int, default=1
        Thickness of trajectory lines.
    centroid_radius : int, default=4
        Radius of centroid circles.
    show_indices : bool, default=True
        Whether to show storm indices.
    font_scale : float, default=0.6
        Font scale for text.
    text_thickness : int, default=2
        Thickness of text.
    max_history_frames : int, default=10
        Maximum number of historical frames to display.
    window : tuple, optional
        Cropping window as (y_start, y_end, x_start, x_end).
    save_path : str, optional
        Path to save the animation.
    fps : int, default=2
        Frames per second for saved video.
    
    Returns:
    --------
    HTML or None
        HTML animation for display, or None if saved to file.
    
    Example:
    --------
    >>> from src.animation import animate_storm_tracking_history
    >>> animate_storm_tracking_history(storms_map_list, max_history_frames=15)
    """
    from src.preprocessing import convert_polygons_to_contours
    
    # Use native DBZ resolution
    H0, W0 = storms_map_list[0].dbz_map.shape
    
    def dbz_to_bgr(dbz_map: np.ndarray, cmap_name: str, vmin_: float, vmax_: float) -> np.ndarray:
        """Convert DBZ map to BGR image."""
        dbz = dbz_map.astype(np.float32)
        dbz = np.clip(dbz, vmin_, vmax_)
        denom = max(vmax_ - vmin_, 1e-6)
        norm = (dbz - vmin_) / denom
        
        cmap_fn = mpl.colormaps.get_cmap(cmap_name)
        rgb = (cmap_fn(norm)[..., :3] * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    
    def get_centroid_from_contour(storm) -> Optional[tuple[float, float]]:
        """Get centroid from storm contour."""
        if not hasattr(storm, "contour") or storm.contour is None:
            return None
        
        cnts = convert_polygons_to_contours([storm.contour])
        if cnts is None or len(cnts) == 0:
            return None
        
        cnt = np.asarray(cnts[0], dtype=np.int32)
        M = cv2.moments(cnt)
        
        if abs(M["m00"]) < 1e-6:
            return None
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return (float(cx), float(cy))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    
    def animate(frame_idx):
        ax.clear()
        ax.axis("off")
        
        current_map = storms_map_list[frame_idx]
        
        # Create DBZ background
        dbz_map = current_map.dbz_map
        if dbz_map.shape != (H0, W0):
            dbz_map = cv2.resize(dbz_map, (W0, H0))
        
        canvas = dbz_to_bgr(dbz_map, cmap, vmin, vmax)
        canvas = np.ascontiguousarray(canvas)
        
        # Calculate history range
        start_frame = max(0, frame_idx - max_history_frames)
        
        # Draw past storm positions (in red)
        for past_idx in range(start_frame, frame_idx):
            past_map = storms_map_list[past_idx]
            
            for storm in past_map.storms:
                cnts = convert_polygons_to_contours([storm.contour])
                if cnts and len(cnts) > 0:
                    cv2.drawContours(canvas, cnts, -1, past_storm_color, 1)
        
        # Draw current storms and their history trajectories
        storm_info_list = []
        
        for storm_idx, storm in enumerate(current_map.storms):
            # Draw current storm contour (in blue)
            cnts = convert_polygons_to_contours([storm.contour])
            if cnts and len(cnts) > 0:
                cv2.drawContours(canvas, cnts, -1, current_storm_color, contour_thickness)
            
            # Get current centroid
            current_centroid = get_centroid_from_contour(storm)
            if current_centroid is None:
                continue
            
            cx, cy = current_centroid
            
            # Draw current centroid
            cv2.circle(canvas, (int(cx), int(cy)), centroid_radius, current_storm_color, -1)
            
            # Draw history trajectory if available
            if hasattr(storm, 'history_movements') and len(storm.history_movements) > 0:
                # Backtrack through history
                end_point = (cx, cy)
                history_len = len(storm.history_movements)
                
                for move_idx, movement in enumerate(storm.history_movements):
                    if frame_idx - (move_idx + 1) < 0:
                        break
                    
                    # Calculate time delta
                    prev_frame_idx = frame_idx - (move_idx + 1)
                    if prev_frame_idx < 0:
                        break
                    
                    prev_frame = storms_map_list[prev_frame_idx].time_frame
                    curr_frame = current_map.time_frame if move_idx == 0 else storms_map_list[frame_idx - move_idx].time_frame
                    dt = (curr_frame - prev_frame).total_seconds() / 3600.0
                    
                    # Movement is in pixels per hour
                    dy, dx = movement[0] * dt, movement[1] * dt
                    
                    start_point = (end_point[0] - dx, end_point[1] - dy)
                    
                    # Draw trajectory line
                    pt1 = (int(end_point[0]), int(end_point[1]))
                    pt2 = (int(start_point[0]), int(start_point[1]))
                    cv2.line(canvas, pt1, pt2, trajectory_color, trajectory_thickness)
                    
                    # Draw past position marker
                    cv2.circle(canvas, pt2, 2, past_storm_color, -1)
                    
                    end_point = start_point
                
                storm_info_list.append((storm_idx, cx, cy, history_len))
            else:
                storm_info_list.append((storm_idx, cx, cy, 0))
        
        # Draw storm indices
        if show_indices:
            for storm_idx, cx, cy, history_len in storm_info_list:
                label = f"{storm_idx}"
                if history_len > 0:
                    label += f" ({history_len})"
                
                # Add text with background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                text_x = int(cx) + centroid_radius + 3
                text_y = int(cy) + 5
                
                # Draw text background
                cv2.rectangle(
                    canvas,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    (255, 255, 255),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    text_thickness
                )
        
        # Apply window cropping if specified
        if window is not None:
            y0, y1, x0, x1 = window
            canvas = canvas[y0:y1, x0:x1]
        
        # Display
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        
        # Title
        time_str = current_map.time_frame.strftime("%Y-%m-%d %H:%M:%S")
        window_str = f" (Window: {window})" if window is not None else ""
        history_str = f" | Showing {min(max_history_frames, frame_idx)} past frames"
        
        ax.set_title(
            f"Storm Tracking History - Frame {frame_idx}\n"
            f"{time_str} | Storms: {len(current_map.storms)}{history_str}{window_str}\n"
            f"Blue=Current | Red=Past | Green=Trajectory | Index(history_length)",
            fontsize=11,
            fontweight="bold"
        )
        
        return ax,
    
    num_frames = len(storms_map_list)
    print(f"Creating storm tracking history animation with {num_frames} frames...")
    print(f"  - Current storms: Blue contours")
    print(f"  - Past positions: Red contours")
    print(f"  - Trajectories: Green lines")
    print(f"  - Max history: {max_history_frames} frames")
    
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval,
        repeat=True,
        blit=False
    )
    
    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        anim.save(save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        return None
    
    plt.close(fig)
    return HTML(anim.to_jshtml())

def build_track_frames_rgb(
    storms_map_list: List,
    model,
    centroid_radius: int = 1,
    line_thickness: int = 2,
    show_indices: bool = True,
    font_scale: float = 0.7,
    text_thickness: int = 2,
    window: Optional[tuple] = None,
    min_track_length: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    interval: int = 200,
    figsize: tuple = (6, 6),
    scale_x: float = 3.0,
    scale_y: float = 3.0,
):
    """
    Build storm track visualization frames and return an animation as HTML.

    Option 1 scaling:
      - scales centroid (x,y) coordinates BEFORE drawing
      - increases canvas size accordingly, so points don't go out of bounds
      - also scales the crop window (if provided)

    Returns:
      HTML(anim.to_jshtml())
    """

    import colorsys
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # ----------------------------
    # Validate frame indices
    # ----------------------------
    start_frame = max(0, start_frame)
    end_frame = end_frame if end_frame is not None else len(storms_map_list)
    end_frame = min(end_frame, len(storms_map_list))

    if start_frame >= end_frame:
        raise ValueError(f"start_frame ({start_frame}) must be < end_frame ({end_frame})")

    # ----------------------------
    # Filter tracks by length
    # ----------------------------
    filtered_tracks = [
        track for track in model.tracker.tracks
        if len(track.storms) >= min_track_length
    ]

    # ----------------------------
    # Fast mapping by object identity
    # ----------------------------
    time_to_frame = {m.time_frame: i for i, m in enumerate(storms_map_list)}
    obj_index = {
        (fi, id(s)): si
        for fi, sm in enumerate(storms_map_list)
        for si, s in enumerate(sm.storms)
    }

    # ----------------------------
    # Assign per-track colors (BGR for OpenCV)
    # ----------------------------
    num_tracks = len(filtered_tracks)
    track_colors = {}
    for track_id in range(num_tracks):
        hue = track_id / max(1, num_tracks - 1) * 0.83
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        track_colors[track_id] = (int(b * 255), int(g * 255), int(r * 255))

    # ----------------------------
    # Build (frame_idx, storm_idx) -> track_id mapping
    # ----------------------------
    storm_to_track = {}
    for track_id, track_hist in enumerate(filtered_tracks):
        for tf, storm_obj in track_hist.storms.items():
            fi = time_to_frame.get(tf, None)
            if fi is None:
                continue
            key = (fi, id(storm_obj))
            if key in obj_index:
                si = obj_index[key]
                storm_to_track[(fi, si)] = track_id

    # ----------------------------
    # Precompute per-frame drawable items
    # Each item: (storm_idx, track_id, (cx, cy))
    # ----------------------------
    frame_items = []
    for fi, sm in enumerate(storms_map_list):
        items = []
        for si, storm in enumerate(sm.storms):
            tid = storm_to_track.get((fi, si), -1)
            if tid < 0:
                continue
            if storm.centroid is None:
                continue
            cx, cy = storm.centroid[::-1]  # (x, y)
            items.append((si, tid, (cx, cy)))
        frame_items.append(items)

    # ----------------------------
    # Create stretched global canvas
    # ----------------------------
    H0, W0 = storms_map_list[0].dbz_map.shape
    H = int(H0 * scale_y)
    W = int(W0 * scale_x)

    canvas_global = np.ones((H, W, 3), dtype=np.uint8) * 255
    last_pos = {}
    frames_rgb = []

    # ----------------------------
    # Build frames (incremental drawing)
    # ----------------------------
    for fi in range(start_frame, end_frame):
        # draw new points and track segments
        for storm_idx, tid, (cx, cy) in frame_items[fi]:
            x = int(cx * scale_x)
            y = int(cy * scale_y)

            color = track_colors.get(tid, (0, 0, 0))

            if tid in last_pos:
                x0, y0 = last_pos[tid]
                cv2.line(canvas_global, (x0, y0), (x, y), color, line_thickness)

            cv2.circle(canvas_global, (x, y), centroid_radius, color, -1)
            last_pos[tid] = (x, y)

        # copy for labels and cropping
        frame_canvas = canvas_global.copy()

        # labels only on current storms in this frame
        if show_indices:
            for storm_idx, tid, (cx, cy) in frame_items[fi]:
                x = int(cx * scale_x)
                y = int(cy * scale_y)

                cv2.putText(
                    frame_canvas,
                    str(storm_idx),
                    (x + centroid_radius + 3, y - centroid_radius),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    text_thickness,
                    cv2.LINE_AA
                )

        # crop window (scaled)
        if window is not None:
            y0, y1, x0, x1 = window
            y0s = int(y0 * scale_y)
            y1s = int(y1 * scale_y)
            x0s = int(x0 * scale_x)
            x1s = int(x1 * scale_x)
            frame_canvas = frame_canvas[y0s:y1s, x0s:x1s]

        # BGR -> RGB for matplotlib
        frames_rgb.append(cv2.cvtColor(frame_canvas, cv2.COLOR_BGR2RGB))

    # ----------------------------
    # Build animation and return HTML
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    im = ax.imshow(frames_rgb[0])

    def _animate(i):
        im.set_data(frames_rgb[i])
        ax.set_title(f"Tracking Frame {start_frame + i}", fontsize=10)
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        _animate,
        frames=len(frames_rgb),
        interval=interval,
        blit=True,
        repeat=True
    )

    plt.close(fig)  # prevent duplicate notebook rendering
    return HTML(anim.to_jshtml())