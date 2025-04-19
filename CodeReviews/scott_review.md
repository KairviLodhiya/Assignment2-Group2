## Style
1. Overall, style was consistent throughout. There were no major issues. All header files had include guards and were set up properly. 
\
## Functionality
1. Early on, element_stiffness.cpp had an error with defining LocalFe. This was fixed after consultation and setting up the appropriate check in Github Actions.
2. Later on, there were multiple `Kokkos::Initialize()` and `Kokkos::Finalize()` calls. To remedy this, I made a main function for unit testing with one call to Kokkos. 
3. LoadVector was originally a .cpp file, but traditionally classes should be defined as .hpp file.
4. LoadVector had capture by value errors when running on the GPU. Additionally, for assembly we had to split the functionality into two separate scripts: one for the GPU and one for the CPU.
5. The original code was not well-optimized for GPU because a lot of storage was restricted to host-only. This was fixed.
6. A driver program was added to either run all unit tests or use the program as intented.
\
## Organization
1. I reorganized the codebase to make it more manageable to find and access files. 
2. We set up Github Actions to properly perform unit tests and check our code.
\
## Does it run?
1. We split assembly into two parts. Checking mesh reader + element stiffness integration and mesh reader + element stiffness and global assembly. This worked out well to ensure each step of the process compiled and outputted the right result. For each assembly process, we have individual unit tests for the functions we wrote and additional tests for the assembly process.
\
## Conclusion
1. Overall, our group did a good job of checking errors, working together, and setting up a well-organized codebase that is functional and works for CPU/GPU parallelization. I could have personally done a better job using branches and PRs, though.
2. Additionally, our team did a good job at having regular meetings and using commit messages as a form of quick review to explain what we fixed and why we fixed it.