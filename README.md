# Cell-segmentation-and-tracking
A school project
##Task 1: Segment and Track Cells
Develop a Python program to segment and track all the cells in the image sequences. This
means the program needs to perform the following steps:
1-1. Segment all the cells and show their contours in the images as overlays. The contour of
each cell should have a unique color and that color should remain the same for the
same cell over time. For each image in a sequence, the program should show the
contours for the cells in that image only.
1-2. Track all the cells over time and show their trajectories (also called tracks, paths,
histories) as overlays. That is, for each image in a sequence, the program should show
for each cell its trajectory up to that time point. The trajectory of a cell is a piecewise
linear curve connecting the centroid positions of the cell, from the time when the cell
first appeared up to the current time point. For each cell, draw its trajectory using the
same color as the contour, for visual consistency. If a cell divides, the two daughter
cells should each get a new color/ID.
