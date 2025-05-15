### Literature Review: Tensor Shape Mismatch Detection

- Gradual Tensor Shape Checking (full version)

  Abstract. Tensor shape mismatch is a common source of bugs in deep
  learning programs. We propose a new type-based approach to detect
  tensor shape mismatches. One of the main features of our approach is
  the best-effort shape inference. As the tensor shape inference problem
  is undecidable in general, we allow static type/shape inference to be
  performed only in a best-effort manner. If the static inference cannot
  guarantee the absence of the shape inconsistencies, dynamic checks are
  inserted into the program. Another main feature is gradual typing, where
  users can improve the precision of the inference by adding appropriate
  type annotations to the program. We formalize our approach and prove
  that it satisfies the criteria of gradual typing proposed by Siek et al. in

paper: https://arxiv.org/pdf/2203.08402

- https://sf.snu.ac.kr/publications/pytea.pdf
- https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf
- https://arxiv.org/pdf/1805.04058
- https://drops.dagstuhl.de/storage/00lipics/lipics-vol166-ecoop2020/LIPIcs.ECOOP.2020.15/LIPIcs.ECOOP.2020.15.pdf
