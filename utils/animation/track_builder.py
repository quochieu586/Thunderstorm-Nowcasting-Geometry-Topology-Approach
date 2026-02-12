from typing import List, Dict, Any, Optional, Hashable, Tuple
from datetime import datetime
import colorsys, json
import os


def to_json_time(t):
    """
    Convert datetime-like objects into JSON-safe strings.
    If it's already safe (int/str/etc.) it returns as-is.
    """
    if isinstance(t, datetime):
        return t.isoformat()
    return t


def build_tracking_store(
    storms_map_list: List,
    model,
    min_track_length: int = 1,
    include_untracked: bool = False,
    include_track_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-friendly tracking store from storms_map_list + model.tracker.tracks

    Output contains:
      - meta (JSON-safe)
      - frames: per-frame storms with (storm_idx, track_id, centroid)
      - tracks: per-track list of points (ordered) + merge/split metadata + per-point event markers

    Notes:
      - "track_id" in JSON is the OUTPUT index (0..n-1) after filtering.
      - "source_track_id" is your StormTrack.id (stable across filtering).
      - split events are marked at splitted_at_frame_idx if possible (exact).
      - merge events are marked at merged_at_frame_idx if a point exists; otherwise at the last point < merged_at_frame_idx.
    """

    # ---------- frame info ----------
    H, W = storms_map_list[0].dbz_map.shape

    raw_time_frames = [m.time_frame for m in storms_map_list]
    json_time_frames = [to_json_time(tf) for tf in raw_time_frames]
    time_to_frame = {tf: i for i, tf in enumerate(raw_time_frames)}

    # ---------- filter tracks ----------
    include_set = set(include_track_ids) if include_track_ids else None
    filtered_tracks = [
        track for track in model.tracker.tracks
        if len(track.storms) >= min_track_length
        and (include_set is None or track.id in include_set)
    ]

    # Map source StormTrack.id -> output track_id
    source_id_to_out_id: Dict[int, int] = {t.id: i for i, t in enumerate(filtered_tracks)}
    # Map object identity -> output track_id (works for merged_to / splitted_from object refs)
    track_obj_to_out_id: Dict[int, int] = {id(t): i for i, t in enumerate(filtered_tracks)}

    def _resolve_ref_to_out_id(ref) -> Optional[int]:
        """
        Resolve merged_to / splitted_from reference into OUTPUT track_id.
        ref can be None | StormTrack object | int (StormTrack.id).
        """
        if ref is None:
            return None
        if isinstance(ref, int):
            return source_id_to_out_id.get(ref, None)

        out = track_obj_to_out_id.get(id(ref), None)
        if out is not None:
            return out

        rid = getattr(ref, "id", None)
        if isinstance(rid, int):
            return source_id_to_out_id.get(rid, None)

        return None

    # ---------- robust storm key ----------
    def _storm_key(storm_obj) -> Hashable:
        """
        Prefer a stable id if storm_obj has .id; else fall back to object identity.
        Must be hashable.
        """
        sid = getattr(storm_obj, "id", None)
        if sid is not None:
            # ensure hashable
            try:
                hash(sid)
                return sid
            except Exception:
                pass
        return id(storm_obj)

    # ---------- index storms in frames: (frame_idx, storm_key) -> storm_idx ----------
    frame_storm_index: Dict[Tuple[int, Hashable], int] = {}
    for fi, sm in enumerate(storms_map_list):
        for si, s in enumerate(sm.storms):
            frame_storm_index[(fi, _storm_key(s))] = si

    # ---------- storm_to_track: (frame_idx, storm_idx) -> output track_id ----------
    storm_to_track: Dict[Tuple[int, int], int] = {}

    for out_tid, track_hist in enumerate(filtered_tracks):
        for tf_raw, storm_obj in track_hist.storms.items():
            fi = time_to_frame.get(tf_raw, None)
            if fi is None:
                continue

            key = (fi, _storm_key(storm_obj))
            si = frame_storm_index.get(key, None)
            if si is None:
                continue

            storm_to_track[(fi, si)] = out_tid

    # ---------- build frames ----------
    frames: List[Dict[str, Any]] = []
    for fi, sm in enumerate(storms_map_list):
        storms_out = []

        for si, storm in enumerate(sm.storms):
            tid = storm_to_track.get((fi, si), -1)

            if (not include_untracked) and tid < 0:
                continue
            if storm.centroid is None:
                continue

            # storm.centroid is (row, col) -> (x, y)
            cx, cy = storm.centroid[::-1]

            storms_out.append({
                "storm_idx": si,
                "track_id": tid,
                "centroid": [float(cx), float(cy)]
            })

        frames.append({
            "frame_idx": fi,
            "time_frame": json_time_frames[fi],
            "storms": storms_out
        })

    # ---------- build tracks: metadata + points ----------
    tracks: List[Dict[str, Any]] = []

    for out_tid, tr in enumerate(filtered_tracks):
        merged_to_out = _resolve_ref_to_out_id(getattr(tr, "merged_to", None))
        splitted_from_out = _resolve_ref_to_out_id(getattr(tr, "splitted_from", None))

        merged_at_raw = getattr(tr, "merged_at", None)
        splitted_at_raw = getattr(tr, "splitted_at", None)

        merged_at_json = to_json_time(merged_at_raw)
        splitted_at_json = to_json_time(splitted_at_raw)

        merged_at_fi = time_to_frame.get(merged_at_raw, None) if merged_at_raw is not None else None
        splitted_at_fi = time_to_frame.get(splitted_at_raw, None) if splitted_at_raw is not None else None

        tracks.append({
            "track_id": out_tid,                 # output id
            "source_track_id": tr.id,            # original StormTrack.id

            "points": [],

            "merged_to": merged_to_out,
            "merged_at": merged_at_json,
            "merged_at_frame_idx": merged_at_fi,

            "splitted_from": splitted_from_out,
            "splitted_at": splitted_at_json,
            "splitted_at_frame_idx": splitted_at_fi,

            "color_rgb": None,                   # filled later
        })

    # Add points (from frames)
    for frame in frames:
        fi = frame["frame_idx"]
        tf_json = frame["time_frame"]

        for s in frame["storms"]:
            tid = s["track_id"]
            if tid < 0:
                continue

            tracks[tid]["points"].append({
                "frame_idx": fi,
                "time_frame": tf_json,
                "storm_idx": s["storm_idx"],
                "centroid": s["centroid"],

                # event markers (consumed by your SVG builder)
                "event": None,                    # "split_from" | "merged_to" | None
                "event_peer_track_id": None,      # output track id of peer
                "event_time_frame": None,         # used for merge fallback annotation
            })

    # ---------- mark split/merge events on points (frame_idx based) ----------
    for t in tracks:
        pts = t["points"]
        if not pts:
            continue

        # Split: try exact frame match (preferred)
        if t["splitted_from"] is not None and t["splitted_at_frame_idx"] is not None:
            split_fi = t["splitted_at_frame_idx"]

            hit = None
            for p in pts:
                if p["frame_idx"] == split_fi:
                    hit = p
                    break

            # If you truly want "exact only", keep only the hit block.
            # This fallback makes splits still visible when the exact frame point is missing.
            if hit is None:
                later = [p for p in pts if p["frame_idx"] > split_fi]
                if later:
                    hit = later[0]

            if hit is not None:
                hit["event"] = "split_from"
                hit["event_peer_track_id"] = t["splitted_from"]

        # Merge: exact frame if exists; else last point before merge frame
        if t["merged_to"] is not None and t["merged_at_frame_idx"] is not None:
            merge_fi = t["merged_at_frame_idx"]

            exact = None
            for p in pts:
                if p["frame_idx"] == merge_fi:
                    exact = p
                    break

            if exact is not None:
                exact["event"] = "merged_to"
                exact["event_peer_track_id"] = t["merged_to"]
            else:
                prev = [p for p in pts if p["frame_idx"] < merge_fi]
                if prev:
                    lastp = prev[-1]
                    lastp["event"] = "merged_to"
                    lastp["event_peer_track_id"] = t["merged_to"]
                    # Keep your UI hint that merge happened later than the point
                    lastp["event_time_frame"] = t["merged_at"]

    # ---------- stable track colors ----------
    num_tracks = len(tracks)
    for t in tracks:
        tid = t["track_id"]
        hue = tid / max(1, num_tracks - 1) * 0.83
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        t["color_rgb"] = [int(r * 255), int(g * 255), int(b * 255)]

    return {
        "meta": {
            "image_shape": [H, W],
            "time_frames": json_time_frames,
            "min_track_length": min_track_length,
            "included_track_ids": include_track_ids,
        },
        "frames": frames,
        "tracks": tracks,
    }

def save_tracking_store_json(store: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(store, f, indent=2)