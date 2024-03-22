\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.6557017132883192 f(x_0) + 0.5253585829846666 f(x_1)
   + 0.4142762515811796 f(x_2) + 0.2500339576085182 f(x_3)
   + 0.4248308924398291 f(x_4)
Subject To
 constr_x_0_b=1: x_0 - 2.0001 x_0>1 >= -1
 constr_x_0_b=0: x_0 - 2.0001 x_0>1 <= 1
 constr_x_1_b=1: x_1 - 2.0001 x_1>1 >= -1
 constr_x_1_b=0: x_1 - 2.0001 x_1>1 <= 1
 constr_x_2_b=1: x_2 - 2.0001 x_2>1 >= -1
 constr_x_2_b=0: x_2 - 2.0001 x_2>1 <= 1
 constr_x_3_b=1: x_3 - 2.0001 x_3>1 >= -1
 constr_x_3_b=0: x_3 - 2.0001 x_3>1 <= 1
 constr_x_4_b=1: x_4 - 2.0001 x_4>1 >= -1
 constr_x_4_b=0: x_4 - 2.0001 x_4>1 <= 1
 R10: x_0 + x_1 + x_2 + x_3 + x_4 = 2
 constr_f(x_0)_b=1: x_0>1 = 1 -> f(x_0) = 0
 constr_f(x_0)_b=0: x_0>1 = 0 -> - x_0 + f(x_0) = 0
 constr_f(x_1)_b=1: x_1>1 = 1 -> f(x_1) = 0
 constr_f(x_1)_b=0: x_1>1 = 0 -> - x_1 + f(x_1) = 0
 constr_f(x_2)_b=1: x_2>1 = 1 -> f(x_2) = 0
 constr_f(x_2)_b=0: x_2>1 = 0 -> - x_2 + f(x_2) = 0
 constr_f(x_3)_b=1: x_3>1 = 1 -> f(x_3) = 0
 constr_f(x_3)_b=0: x_3>1 = 0 -> - x_3 + f(x_3) = 0
 constr_f(x_4)_b=1: x_4>1 = 1 -> f(x_4) = 0
 constr_f(x_4)_b=0: x_4>1 = 0 -> - x_4 + f(x_4) = 0
Bounds
 x_0 <= 2
 x_1 <= 2
 x_2 <= 2
 x_3 <= 2
 x_4 <= 2
Binaries
 x_0>1 x_1>1 x_2>1 x_3>1 x_4>1
Generals
 x_0 x_1 x_2 x_3 x_4
End
