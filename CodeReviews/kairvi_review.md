### Some Problems faced while doing assignment
#### Kokkos Initialization and Finalization Overlap:
Repeated initialization/finalization of Kokkos in multiple test files caused runtime issues. This was resolved by centralizing all Kokkos lifecycle management within a single main.cpp file that runs both tests and driver functionality.

#### Incorrect Mesh Generation:
The auto-generated meshes using Gmsh contained mixed element types or improper formatting, leading to the error: "Mixed element types not supported." This was fixed by adjusting the Gmsh .geo file to consistently produce triangular elements and verifying each mesh with diagnostic scripts.

#### Timings Not Captured in CSV:
Early attempts to capture runtime data in timings.csv failed due to incorrect parsing or inconsistent console output. This was fixed by updating the output format in main.cpp and carefully filtering lines with grep and awk in the run_timings.sh script.

#### Git Push Conflicts and Workflow Breakdown:
Collaborative development was initially planned using branches and pull requests, but frequent non-fast-forward and merge conflict errors led to working directly on the main branch. This reduced our ability to conduct code reviews, but inline comments were added wherever peer modifications were made.

##### PS: Sorry for uploading this late, just realized this was to be done!
