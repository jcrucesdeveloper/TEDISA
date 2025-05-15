### Literature Review: Tensor Shape Mismatch Detection

- [Gradual Tensor Shape Checking (full version)](https://arxiv.org/pdf/2203.08402)

  Tensor shape mismatch is a common source of bugs in deep
  learning programs. We propose a new type-based approach to detect
  tensor shape mismatches. One of the main features of our approach is
  the best-effort shape inference. As the tensor shape inference problem
  is undecidable in general, we allow static type/shape inference to be
  performed only in a best-effort manner. If the static inference cannot
  guarantee the absence of the shape inconsistencies, dynamic checks are
  inserted into the program. Another main feature is gradual typing, where
  users can improve the precision of the inference by adding appropriate
  type annotations to the program. We formalize our approach and prove
  that it satisfies the criteria of gradual typing proposed by Siek et al. paper: https://arxiv.org/pdf/2203.08402

- [A Static Analyzer for Detecting Tensor Shape Errors in Deep Neural Network Training Code](https://sf.snu.ac.kr/publications/pytea.pdf)

  We present an automatic static analyzer PyTea that detects tensorshape errors in PyTorch code. The tensor-shape error is critical in
  the deep neural net code; much of the training cost and intermediate results are to be lost once a tensor shape mismatch occurs in
  the midst of the training phase. Given the input PyTorch source,
  PyTea statically traces every possible execution path, collects tensor
  shape constraints required by the tensor operation sequence of the
  path, and decides if the constraints are unsatisfiable (hence a shape
  error can occur). PyTea’s scalability and precision hinges on the
  characteristics of real-world PyTorch applications: the number of
  execution paths after PyTea’s conservative pruning rarely explodes
  and loops are simple enough to be circumscribed by our symbolic
  abstraction. We tested PyTea against the projects in the official
  PyTorch repository and some tensor-error code questioned in the
  StackOverflow. PyTea successfully detects tensor shape errors in
  these codes, each within a few seconds.

https://sf.snu.ac.kr/publications/pytea.pdf

- https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf
- https://arxiv.org/pdf/1805.04058
- https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf
