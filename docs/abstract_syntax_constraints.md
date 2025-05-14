Constraint
𝑐 -> 𝑐 ∧ 𝑐
| 𝑐 ∨ 𝑐
| ¬𝑐
| 𝑒𝑏
| 𝑒𝑏
| 𝑒 = 𝑒
| 𝑒𝑛 < 𝑒𝑛
| ∀𝛼_n ∈ [𝑒𝑛, 𝑒𝑛].𝑐 (𝑐 is true forall integer 𝛼𝑛
in the interval)

Value Expr
𝑒 → 𝑒𝑠 | 𝑒𝑛 | 𝑒𝑏
(shape, number, or boolean)
Shape Expr
𝑒𝑠 → ( 𝑒𝑛, · · · ,𝑒𝑛) (tensor shape)
| 𝛼𝑠 (unknown shape)
| 𝑒𝑠[ 𝑒𝑛:𝑒𝑛] (shape slicing)
| 𝑒𝑠@ 𝑒𝑠 (shape concat)
Number Expr
𝑒𝑛 → 𝑛 (const number)
| 𝛼𝑛 (unknown number)
| 𝑒𝑛 bop 𝑒𝑛 (binary operator)
| rank ( 𝑒𝑠) (rank of shape)
| 𝑒𝑠[ 𝑒𝑛] (𝑒𝑛-th dimension of shape 𝑒𝑠 )
|
Î𝑒𝑠 (number of elements in
tensor of shape 𝑒𝑠 )
bop → + | - | \* | · · ·
Boolean Expr
𝑒𝑏 → True | False
| 𝛼𝑏
(unknown boolean)
| 𝑒𝑏 ∧ 𝑒𝑏
(conjunction)
| 𝑒𝑏 ∨ 𝑒𝑏
(disjuction)
| ¬𝑒𝑏
(negation)
| 𝑒 = 𝑒 (equality)
| 𝑒𝑛 < 𝑒𝑛 (less than)

Constraint
c -> c ∧ c
| c ∨ c
| not c
| bool
| num = num
| num < num
| 

Shape Expr
shape -> (num_1, ... , num_n) (tensor shape)

Number Expr
num -> x (const number)
| x1 bop x2
| dim(s) (dimension of a shape)
| s[num] (n-th dimension of shape s)
| pitatoria(s) (number of elements in that shape)
