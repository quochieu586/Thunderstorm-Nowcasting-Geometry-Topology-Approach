import json

COLOR_LEGEND_PATH = 'data/legend/color_dbz.json'

with open(COLOR_LEGEND_PATH) as f:
    list_color = json.load(f)

sorted_color = sorted({tuple(color[1]): color[0] for color in list_color}.items(), key=lambda item: item[1])