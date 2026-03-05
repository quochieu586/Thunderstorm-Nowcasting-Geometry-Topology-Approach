import sys
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np

from utils.evaluation_models import ModelSetting, SensitivityAnalysisResults, create_model, create_identifier

DATASET_PATH = "data/numpy_grid/"

def run_models_evaluation(dataset: str, analysis: ModelSetting) -> SensitivityAnalysisResults:
    """
        Main entries to run sensitivity analysis. Phases include:
            1. Load dataset and configure model with given parameters
            2. Identify storms in each frame
            3. Perform and evaluate nowcasting across time
            4. Evaluation of tracking performance
    """
    # Phase 1: Load dataset and configure model
    from src.preprocessing import read_numpy_grid

    source_path = os.path.join(DATASET_PATH, dataset)
    img_paths = [os.path.join(source_path, img_name) for img_name in sorted(os.listdir(source_path)) if img_name.endswith('.npy')]
    dbz_maps: list[tuple[np.ndarray, datetime]] = []

    for path in tqdm(img_paths, desc="Processing images and detecting storms"):
        file_name = path.split("/")[-1].split(".")[0]

        time_frame = datetime.strptime(file_name[4:19], "%Y%m%d_%H%M%S")
        img = read_numpy_grid(path)
        dbz_maps.append((img, time_frame))

    # Phase 2: Identify storms
    storms_maps = []
    for idx, (dbz_map, time_frame) in tqdm(list(enumerate(dbz_maps)), desc="Identifying storms in each frame"):
        storms_map = analysis.model.identify_storms(dbz_map, time_frame, map_id=f"time_{idx}", threshold=35, filter_area=50)
        storms_maps.append(storms_map)

    # Phase 3: Evaluate Nowcasting across time
    from src.cores.metrics import PredictionBenchmarkModel

    # Evaluate for 3-step ahead prediction
    ours_model_evaluation_3 = PredictionBenchmarkModel()
    temp_storm_map = storms_maps

    model = analysis.model.copy()
    SLOW_START_STEPS = 3
    PREDICT_FORWARD_STEPS = 3

    for i in range(SLOW_START_STEPS):
        model.processing_map(temp_storm_map[i])  # Warm-up phase

    for curr_map, future_map in tqdm(list(zip(temp_storm_map[SLOW_START_STEPS:], temp_storm_map[PREDICT_FORWARD_STEPS + SLOW_START_STEPS:])), desc="Predicting precipitation maps"):
        # Predict map using current data
        dt_seconds = (future_map.time_frame - model.storms_maps[-1].time_frame).total_seconds()
        predicted_map = model.forecast(dt_seconds)
        ours_model_evaluation_3.evaluate_predict(future_map, predicted_map)
        model.processing_map(curr_map)  # Update model with the current map

    # Evaluate for 5-step ahead prediction
    ours_model_evaluation_5 = PredictionBenchmarkModel()
    temp_storm_map = storms_maps

    model = analysis.model.copy()
    SLOW_START_STEPS = 5
    PREDICT_FORWARD_STEPS = 5

    for i in range(SLOW_START_STEPS):
        model.processing_map(temp_storm_map[i])  # Warm-up phase

    for curr_map, future_map in tqdm(list(zip(temp_storm_map[SLOW_START_STEPS:], temp_storm_map[PREDICT_FORWARD_STEPS + SLOW_START_STEPS:])), desc="Predicting precipitation maps"):
        # Predict map using current data
        dt_seconds = (future_map.time_frame - model.storms_maps[-1].time_frame).total_seconds()
        predicted_map = model.forecast(dt_seconds)
        ours_model_evaluation_5.evaluate_predict(future_map, predicted_map)
        model.processing_map(curr_map)  # Update model with the current map

    # Phase 4: Evaluate tracking performance
    from src.cores.metrics import PostEventClustering, linear_tracking_error

    # 4.1 Object consistency first
    object_consistency_scores = []
    for track in model.tracker.tracks:
        areas = [storm.contour.area for storm in track.storms.values()]
        area_changes = [abs(areas[i] - areas[i - 1]) / areas[i - 1] for i in range(1, len(areas)) if areas[i - 1] != 0]
        object_consistency_scores.append(np.mean(area_changes) if area_changes else 0)

    object_consistency_score = np.mean(object_consistency_scores)

    # 4.2 Mean duration of tracked objects
    mean_duration = np.mean([len(track.storms) for track in model.tracker.tracks])

    # 4.3 Linear RMSE
    linear_error_lsts = [linear_tracking_error(storm.history_movements[:-1]) ** 2 for storms_map in storms_maps 
                                                                    for storm in storms_map.storms if len(storm.history_movements) >= mean_duration]
    linear_rmse = np.sqrt(np.mean(linear_error_lsts))

    # 4.4 Optimal tracking evaluation
    centroids = []
    clusters_assigned = []

    FIRST_TIME_FRAME = dbz_maps[0][1]

    def _convert_time_frame_to_seconds(time_frame: datetime) -> float:
        return time_frame.timestamp() - FIRST_TIME_FRAME.timestamp()

    for track_history in model.tracker.tracks:
        track_centroids = [(storm.centroid[0], storm.centroid[1], _convert_time_frame_to_seconds(time_frame)) for time_frame, storm in track_history.storms.items()]
        centroids.extend(track_centroids)
        clusters_assigned.extend([track_history.id] * len(track_centroids))
    
    centroids = np.array(centroids)

    postevent_analysis = PostEventClustering(centroids, max_window_time=600, spatial_distance_threshold=50)
    reassigned_clusters_centers = postevent_analysis.fit_transform(num_clusters=len(model.tracker.tracks), clusters_assigned=clusters_assigned, max_epochs=50)

    merged_clusters_assigned = [postevent_analysis.clusters_merged_dict.get(i, i) for i in clusters_assigned]
    score_lst = [1 if merged_clusters_assigned[i] == reassigned_clusters_centers[i] else 0.5 if reassigned_clusters_centers[i] != -1 else 0 for i in range(len(clusters_assigned))]

    optimal_tracking_score = np.mean(score_lst)

    return SensitivityAnalysisResults(
        pod_3=np.mean(ours_model_evaluation_3.pods),
        far_3=np.mean(ours_model_evaluation_3.fars),
        csi_3=np.mean(ours_model_evaluation_3.csis),
        pod_5=np.mean(ours_model_evaluation_5.pods),
        far_5=np.mean(ours_model_evaluation_5.fars),
        csi_5=np.mean(ours_model_evaluation_5.csis),
        object_consistency=object_consistency_score,
        mean_duration=mean_duration,
        linear_rmse=linear_rmse,
        optimal_tracking=optimal_tracking_score
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thunderstorm Nowcasting Geometry Approach")
    parser.add_argument("--dataset", type=str, default="KARX", help="Dataset to run the evaluation on")
    parser.add_argument("--model", type=str, default="ours", help="Choosing one model to run")
    parser.add_argument("--max-velocity", type=int, default=100, help="Maximum velocity for the model")
    parser.add_argument("--identification-method", type=str, default="morphology", help="Object identification method, one of ['simple', 'morphology', 'hypothesis', 'cluster']")

    print("-" * 50)
    print("Starting evaluation with the following configuration:")
    print(f"Dataset: {parser.parse_args().dataset}")
    print(f"Model: {parser.parse_args().model}")
    print(f"Max Velocity: {parser.parse_args().max_velocity} km/h")
    print(f"Identification Method: {parser.parse_args().identification_method}")

    print("-" * 50)

    identifier = create_identifier(identification_method=parser.parse_args().identification_method)
    model = create_model(model_name=parser.parse_args().model, identifier=identifier, max_velocity=parser.parse_args().max_velocity)

    model_setting = ModelSetting(
        identification_method=parser.parse_args().identification_method,
        model_name=parser.parse_args().model,
        model=model,
        max_velocity=int(parser.parse_args().max_velocity)
    )

    results = run_models_evaluation(dataset=parser.parse_args().dataset, analysis=model_setting)
    results.print_result()
