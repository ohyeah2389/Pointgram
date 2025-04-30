# Pointgram
Pointgram is a tool that allows the user to perform manual feature extraction across multiple input images, store the features as a project file, and call COLMAP (through PyCOLMAP) to use those features to solve the input images' camera's intrinsic and extrinsic properties, as well as the locations of the points. This is a common technique in vehicle modeling and other artistic reverse engineering applications where photogrammetry through automatic feature extraction may not be possible, such as with a reference set of objects that are the same shape, but different colors, or with different backgrounds.

A historical example of this functionality is the program Autodesk ImageModeler 2009, which is still commonly used, especially in the hobbyist or amatuer vehicle modeling scene. Some features of ImageModeler are out of scope for this project, such as the in-program modeling tools, but its feature set is a general target for this program's functionality.

Currently, the program is capable of all steps from image export to reconstructed scene export in GLTF format, but some key features are missing:

- Point-to-point distance definition for rescaling
- Three-point plane picking for coordinate system definition

The program has been tested with Python 3.9 and PyCOLMAP 3.11.1. It uses PySide6 v6.9.0 for UI operations, along with the Tango icons included in the repository.