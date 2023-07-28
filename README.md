# ExtendableFEM
High Level API Finite Element Methods based on ExtendableGrids and ExtendableFEMBase. 
It offers a ProblemDescription interface, that basically involves assigning Unknowns
and operators. Such operator usually stem from a weak formulation of the problem
and mainly consist of three types that can be customized via kernel functions:

- BilinearOperator,
- LinearOperator,
- NonlinearOperator (that automatically assemble Newton method by automatic differentiation)


