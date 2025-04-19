# Major issues and remedies done
1. Initialization of Kokkos was called upon too many times in the test files, which was dealth with by using a single main file for initialization and finalization.
2. Couple of files had GPU parallelization looking into the host memory errors. It was fixed ans should not be an issue now.
3. I would say the codes could use more comments and documentations inside them to describe the parameters and functions used.
4. Creating branches and pushing into them to create pull requests was giving us a lot of git ahead and behind errors so we opted to do everything on the main branch. Because, of that we could not do much of code reviews. Instead, I have commented on my friends codes where I have made required changes.