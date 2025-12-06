import numpy as np
import pandas as pd
from shapely.affinity import translate
from sklearn.cluster import KMeans
from dataclasses import dataclass, field

from src.cores.base import StormsMap
from src.tracking import BaseMatcher
from src.models.base.tracker import UpdateType
from .particle_matcher import MatchedStormInfo, ParticleMatcher
from .storm import DbzStormsMap

from .default import DEFAULT_COARSE_MATCHING_THRESHOLD, DEFAULT_FINE_MATCHING_THRESHOLD

@dataclass
class UpdateInfo:
    prev_storm_order: int
    curr_storm_order: int
    update_type: UpdateType
    velocity: np.ndarray = field(default=None)

@dataclass
class CellSubset:
    """
    Keep track of matched storms between 2 frames. Each storm cell in source is matched with at least one storm cell in target and vice versa.
    """
    sources: set[int]      # list of indices of source storms
    targets: set[int]      # list of indices of target storms

    def add_cell(self, source: int = None, target: int = None):
        """
        add a source/target storm index to the subset.
        """
        if source is not None:
            self.sources.add(source)
        if target is not None:
            self.targets.add(target)
    
    def merge_subset(self, other: "CellSubset"):
        """
        merge another subset into this one.
        """
        self.sources = self.sources.union(other.sources)
        self.targets = self.targets.union(other.targets)

    def contains(self, storm_order: int, is_source: bool=True) -> bool:
        """
        check if the cell contains the given source/target storm index.
        """
        return (storm_order in self.sources) if is_source else (storm_order in self.targets)

class StormMatcher(BaseMatcher):
    max_velocity: float
    particle_matcher: ParticleMatcher
    weights: tuple[float, float]

    def __init__(self, max_velocity: float, weights: tuple[float, float]):
        self.max_velocity = max_velocity
        self.weights = weights
        self.particle_matcher = ParticleMatcher()
    
    def _construct_disparity_matrix(self, object_lst1, object_lst2):
        pass

    def _resolve_displacement(self, displacements: list[float, tuple[float, float]]):
        """
        Resolve the displacement from a list of (score, displacement) pairs. Return the weighted average of the displacements.
        """
        # return np.array(max(displacements, key=lambda x: x[0])[1])
        weights = np.array([score for score, _ in displacements])
        total_weight = np.sum(weights)

        displacements_array = np.array([np.array(displacement) for _, displacement in displacements])

        return np.sum(displacements_array * (weights[:, np.newaxis] / total_weight), axis=0)

    def match_storms(
            self, storms_map_lst_1: DbzStormsMap, storms_map_lst_2: DbzStormsMap,
            coarse_threshold: float = DEFAULT_COARSE_MATCHING_THRESHOLD, fine_threshold: float = DEFAULT_FINE_MATCHING_THRESHOLD
        ) -> list[UpdateInfo]:
        """
        Match storms between 2 time frame.

        Args:
            storm_map1 (StormsMap): storm map in the 1st frame.
            storm_map2 (StormsMap): storm map in the 2nd frame.
        
        Returns:
            tuple[np.ndarray, list, list]:
                assignments (nd.ndarray): list of pairs of corresponding id of 2 storms.
                probability_matrix (list): list of score of corresponding assignment.
                displacements (list): list of displacement of corresponding assignment.
        """
        # step 1: particle matching
        dt = (storms_map_lst_2.time_frame - storms_map_lst_1.time_frame).seconds / 3600
        maximum_displacement = self.max_velocity * dt

        prev_storms_lst = storms_map_lst_1.storms
        curr_storms_lst = storms_map_lst_2.storms

        matched_info_list = self.particle_matcher.match_storms(
            prev_storms_list=prev_storms_lst, curr_storms_list=curr_storms_lst, 
            weights=self.weights, maximum_displacement=maximum_displacement, matching_threshold=coarse_threshold
        )

        # step 2: resolve subset assignments, derive coarsed motion vectors
        ## 2.1 create subsets
        subset_list: list[CellSubset] = []
        for info in matched_info_list:
            prev_storm_order = int(info.prev_storm_id.split("_")[-1])
            curr_storm_order = int(info.curr_storm_id.split("_")[-1])

            prev_subset = None
            curr_subset = None

            # find if there is existing subset containing prev_idx or curr_idx
            for subset in subset_list:
                if subset.contains(prev_storm_order, is_source=True):
                    prev_subset = subset
                if subset.contains(curr_storm_order, is_source=False):
                    curr_subset = subset
            
            # case 1: both are not belonged to any subset => create new subset
            if not prev_subset and not curr_subset:
                # create new subset
                new_subset = CellSubset(sources={prev_storm_order}, targets={curr_storm_order})
                subset_list.append(new_subset)
                continue
                
            # case 2: only one of them is belonged to a subset => add the other index to that subset
            if prev_subset and not curr_subset:
                prev_subset.add_cell(target=curr_storm_order)
                continue
            if not prev_subset and curr_subset:
                curr_subset.add_cell(source=prev_storm_order)
                continue
                
            # case 3: both are belonged to different subsets => merge the 2 subsets
            if prev_subset != curr_subset:
                prev_subset.merge_subset(curr_subset)
                subset_list.remove(curr_subset)
        
        ## 2.2 resolve the subsets
        justified_matched_info_list: list[MatchedStormInfo] = []
        for subset in subset_list:
            sub_prev_storms = [storms_map_lst_1.storms[idx] for idx in subset.sources]
            sub_curr_storms = [storms_map_lst_2.storms[idx] for idx in subset.targets]

            # perform matching within each subset with fine threshold
            justified_matched_info_list.extend(
                self.particle_matcher.match_storms(
                    prev_storms_list=sub_prev_storms, curr_storms_list=sub_curr_storms, 
                    weights=self.weights, maximum_displacement=maximum_displacement, matching_threshold=fine_threshold
                )
            )
        
        # Step 3. resolve split and merge
        ## Mapping from current -> list of matched previous storms
        mapping_curr = {i: [] for i in range(len(storms_map_lst_2.storms))}

        for info in justified_matched_info_list:
            prev_storm_order = int(info.prev_storm_id.split("_")[-1])
            curr_storm_order = int(info.curr_storm_id.split("_")[-1])

            mapping_curr[curr_storm_order].append((
                prev_storm_order,
                info.curr_score,
                info.prev_score,
                info.derive_displacement()
            ))
        
        ## Checking the inheritance status
        inherited = {curr_idx: {
                'parent': None,
                'merged_list': [],
                'is_split': False,
                'coarse_displacement': None
            } for curr_idx in range(len(storms_map_lst_2.storms))}   # curr_idx -> parent info
        
        parent_dict = {i: [] for i in range(len(storms_map_lst_1.storms))}    # keep track of which curr storms inherit from which prev storms

        for curr_idx, parent_list in mapping_curr.items():
            # case 1: no parent => new storm
            if len(parent_list) == 0:
                continue
            
            # case 2: only one parent => normal matched storm
            if len(parent_list) == 1:
                parent_idx, _, prev_score, displacement = parent_list[0]
                inherited[curr_idx]['parent'] = parent_idx
                inherited[curr_idx]['coarse_displacement'] = displacement
                parent_dict[parent_idx].append((curr_idx, prev_score))
                continue
            
            # case 3: merged storm
            parent_idx, _, prev_score, _ = max(parent_list, key=lambda x: x[1])
            ## 3.1 record parent info. Parent -> storm with highest curr_score
            inherited[curr_idx]['parent'] = parent_idx
            parent_dict[parent_idx].append((curr_idx, prev_score))
            inherited[curr_idx]['merged_list'] = [item[0] for item in parent_list if item[0] != parent_idx]

            ## 3.2 resolve coarse displacement
            displacements = [(curr_score, motion_vector) for _, curr_score, _, motion_vector in parent_list]
            inherited[curr_idx]['coarse_displacement'] = self._resolve_displacement(displacements)
        
        ## resolve split storms
        for _, children in parent_dict.items():
            if len(children) <= 1:
                continue
            # sort by parent_score descending
            children_sorted = sorted(children, key=lambda x: x[1], reverse=True)

            # first keeps real ID, others become virtual (split)
            for curr_idx, _ in children_sorted[1:]:
                inherited[curr_idx]['is_split'] = True
        
        # step 4: justify the motion vectors using TREC-based method
        coarse_displacements = [inherited[curr_idx]['coarse_displacement'] for curr_idx in range(len(storms_map_lst_2.storms))]
        fine_motions = storms_map_lst_2.estimate_motion_vector_backtrack(storms_map_lst_1, coarse_displacements, 
                                                               max_velocity=self.max_velocity)

        for curr_idx in range(len(storms_map_lst_2.storms)):
            inherited[curr_idx]['fine_motion'] = fine_motions[curr_idx]
        
        # step 5: prepare update info
        update_list: list[UpdateInfo] = []

        for curr_idx, info in inherited.items():
            parent_idx = info['parent']
            is_split = info['is_split']
            merged_list = info['merged_list']
            velocity = info['fine_motion']

            if parent_idx is None:
                # new storm
                update_list.append(UpdateInfo(
                    prev_storm_order=-1,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.NEW
                ))
                continue
            
            if is_split:
                # splitted storm
                update_list.append(UpdateInfo(
                    prev_storm_order=parent_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.SPLITTED,
                    velocity=velocity
                ))
            else:
                # matched storm
                update_list.append(UpdateInfo(
                    prev_storm_order=parent_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MATCHED,
                    velocity=velocity
                ))

            ## update merged storms
            for merged_idx in merged_list:
                update_list.append(UpdateInfo(
                    prev_storm_order=merged_idx,
                    curr_storm_order=curr_idx,
                    update_type=UpdateType.MERGED
                ))
            
        return update_list