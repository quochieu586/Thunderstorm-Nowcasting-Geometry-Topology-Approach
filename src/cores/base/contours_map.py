from dataclasses import dataclass
from datetime import datetime
from matplotlib import pyplot as plt
from .contours import StormObject

THRESHOLD_DBZ = 30

@dataclass
class StormsMap:
    storms: list[StormObject]
    time_frame: datetime

    def plot(
        self,
        show_particles=True,
        show_vectors=False,
        title="Shape Vector Storms",
        cmap="HomeyerRainbow",
    ):
        """
            Plots the storms on top of the dbz_map if available.
            Only works if the StormsMap instance has a dbz_map attribute.
        """
        if not hasattr(self, 'dbz_map'):
            raise AttributeError("This StormsMap instance dbz_map attribute is not found. Perhaps this storms tracking pipeline doesn't use a dbz_map?")

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.dbz_map, cmap=cmap, origin="upper", vmin=-8, vmax=75)
        for storm in self.storms:
            storm.plot_on(ax)
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Reflectivity (dBZ)")
        plt.tight_layout()
        plt.show()