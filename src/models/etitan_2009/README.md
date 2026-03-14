# ETitan (2009)
![ETitan 2009](../../../docs/images/experimental_etitan.png)
---
### **Model Workflow**
1. **Generating a motion vector field**: ETITAN employs TREC algorithm proposed in Rinehart and Garvey (1978) to compute a motion vector field $V_{TREC}$ that represents the precipitation shift between two frames. In particular, $V_{TREC}$ is computed as follows: 
    - (1) partition the first reflectivity map into fixed spatial blocks. 
    - (2) for each block, define a corresponding search region in the subsequent map by extending the block with a buffer determined by the maximum allowable displacement. 
    - (3) estimate the displacement vector that maximizes the cross-correlation.
2. **Matching by overlapping technique**: For each storm in the first map, forecast its position in the subsequent map using $V_{TREC}$ (introduced in Step 4). If the predicted storm overlaps with any storm in the later map, compute the association score as:
$$
f = f_1 + f_2,\ \text{where }
\begin{cases}
f_1 = \dfrac{\text{the overlap area}}{\text{storm area in the first map}} \\[8pt]
f_2 = \dfrac{\text{the overlap area}}{\text{storm area in the second map}}
\end{cases}
$$
If $f>0.5$, these storms are considered matched.
3. **Hungarian matching with dynamic constraint**: Since a storm at two consecutive scans may not overlap much, a Hungarian matching scheme similar to that used in TITAN is adopted. Instead of applying a fixed maximum velocity constraint, a dynamic threshold is introduced to tolerate random centroid displacements induced by the threshold-based identification. Specifically, the maximum allowable speed is defined as follows: 
    - (1) $100~km~h^{-1}$ for storm $areas<300~km^{2}$. 
    - (2) $150~km~h^{-1}$ for areas between 300 and $500~km^{2}$. 
    - (3) $200~km~h^{-1}$ for $areas>500~km^{2}$.
4. **TREC-based motion estimation**: For each storm in the first map, determine its motion by averaging value of $V_{TREC}$ within the area of the identified storm.

### Experimental Notebook
[View Experimental Notebook](../../../experimental_notebooks/etitan_model.ipynb)