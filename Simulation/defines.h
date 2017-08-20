#pragma once

#define MESH_WIDTH  8 // the overall simulation mesh dimensions
#define MESH_HEIGHT 8
#define MESH_DEPTH  8

#define GRID_SIZE 64 // the dimension of the grid which stores the particles

#define FETCH(t, i) t[i] // macro to retrieve element from an array

#define SMALL 1E-12f // a small number, used for comparison purposes