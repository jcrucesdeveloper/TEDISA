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
  that it satisfies the criteria of gradual typing proposed by Siek et al.

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

- [Static Analysis of Shape in TensorFlow Programs](https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf)

  Machine learning has been widely adopted in diverse science and engineering domains, aided by
  reusable libraries and quick development patterns. The TensorFlow library is probably the bestknown representative of this trend and most users employ the Python API to its powerful back-end.
  TensorFlow programs are susceptible to several systematic errors, especially in the dynamic typing
  setting of Python. We present Pythia, a static analysis that tracks the shapes of tensors across
  Python library calls and warns of several possible mismatches. The key technical aspects are a close
  modeling of library semantics with respect to tensor shape, and an identification of violations and
  error-prone patterns. Pythia is powerful enough to statically detect (with 84.62% precision) 11 of
  the 14 shape-related TensorFlow bugs in the recent Zhang et al. empirical study – an independent
  slice of real-world bugs

- [Ariadne: Analysis for Machine Learning Programs](https://arxiv.org/pdf/1805.04058)

  Machine learning has transformed domains like vision and
  translation, and is now increasingly used in science, where
  the correctness of such code is vital. Python is popular for
  machine learning, in part because of its wealth of machine
  learning libraries, and is felt to make development faster;
  however, this dynamic language has less support for error
  detection at code creation time than tools like Eclipse. This
  is especially problematic for machine learning: given its statistical nature, code with subtle errors may run and produce
  results that look plausible but are meaningless. This can
  vitiate scientific results. We report on Ariadne: applying a
  static framework, WALA, to machine learning code that uses
  TensorFlow. We have created static analysis for Python, a
  type system for tracking tensors—Tensorflow’s core data
  structures—and a data flow analysis to track their usage. We
  report on how it was built and present some early results

- [Static Analysis of Shape in TensorFlow Programs](https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf)

  Machine learning has been widely adopted in diverse science and engineering domains, aided by
  reusable libraries and quick development patterns. The TensorFlow library is probably the bestknown representative of this trend and most users employ the Python API to its powerful back-end.
  TensorFlow programs are susceptible to several systematic errors, especially in the dynamic typing
  setting of Python. We present Pythia, a static analysis that tracks the shapes of tensors across
  Python library calls and warns of several possible mismatches. The key technical aspects are a close
  modeling of library semantics with respect to tensor shape, and an identification of violations and
  error-prone patterns. Pythia is powerful enough to statically detect (with 84.62% precision) 11 of
  the 14 shape-related TensorFlow bugs in the recent Zhang et al. empirical study – an independent
  slice of real-world bugs.
