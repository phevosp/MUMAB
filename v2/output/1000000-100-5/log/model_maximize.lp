\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.5206052993035822 f(x_0) + 0.6834181098435286 f(x_1)
   + 0.2577338128573314 f(x_2) + 0.7310320519268202 f(x_3)
   + 0.258452720973855 f(x_4) + 0.310683395688297 f(x_5)
   + 0.617564943215847 f(x_6) + 0.5618168886469063 f(x_7)
   + 0.716799864293105 f(x_8) + 0.7347096935117086 f(x_9)
   + 0.4732342178350228 f(x_10) + 0.5012778211293274 f(x_11)
   + 0.5152825159609603 f(x_12) + 0.5886202267014375 f(x_13)
   + 0.4518199953360321 f(x_14) + 0.6226619065891824 f(x_15)
   + 0.2748639352976192 f(x_16) + 0.3681809262993423 f(x_17)
   + 0.3339404027202397 f(x_18) + 0.4840707284592105 f(x_19)
   + 0.3623928223051486 f(x_20) + 0.3879443950890121 f(x_21)
   + 0.3627915837997172 f(x_22) + 0.4141288985049247 f(x_23)
   + 0.455686772775112 f(x_24) + 0.5132672773007225 f(x_25)
   + 0.5234884893364415 f(x_26) + 0.2541691941062165 f(x_27)
   + 0.3515341090550406 f(x_28) + 0.4885919657576224 f(x_29)
   + 0.6713314889921564 f(x_30) + 0.3882311842020765 f(x_31)
   + 0.5262997873585997 f(x_32) + 0.5744629007438079 f(x_33)
   + 0.7355603732769918 f(x_34) + 0.4819072176807733 f(x_35)
   + 0.4723797077885775 f(x_36) + 0.2695188590225934 f(x_37)
   + 0.379549083838789 f(x_38) + 0.4919023554271987 f(x_39)
   + 0.6455535404432673 f(x_40) + 0.6344181887878134 f(x_41)
   + 0.5716214321533402 f(x_42) + 0.3206887009716797 f(x_43)
   + 0.4976558400511513 f(x_44) + 0.5356386658259252 f(x_45)
   + 0.5144591276262228 f(x_46) + 0.2836293641581864 f(x_47)
   + 0.5143905187126965 f(x_48) + 0.7090171699241788 f(x_49)
   + 0.553391649690502 f(x_50) + 0.6806390525482808 f(x_51)
   + 0.3453519189048067 f(x_52) + 0.4575213147289947 f(x_53)
   + 0.5664926445203314 f(x_54) + 0.4979581716743964 f(x_55)
   + 0.4922837092237106 f(x_56) + 0.5038351700272283 f(x_57)
   + 0.4410549677418432 f(x_58) + 0.2513761586917698 f(x_59)
   + 0.4243271372588004 f(x_60) + 0.3946783845995821 f(x_61)
   + 0.3454877089348328 f(x_62) + 0.2733982925471551 f(x_63)
   + 0.352100362578767 f(x_64) + 0.3409775284577377 f(x_65)
   + 0.3359075262881225 f(x_66) + 0.3680075357029801 f(x_67)
   + 0.5007064699984143 f(x_68) + 0.6697796341723596 f(x_69)
   + 0.2549080942166873 f(x_70) + 0.2931872110400958 f(x_71)
   + 0.3011428202922743 f(x_72) + 0.373023573879997 f(x_73)
   + 0.2977244535342077 f(x_74) + 0.6592675726808754 f(x_75)
   + 0.6340250419384259 f(x_76) + 0.5572923720135559 f(x_77)
   + 0.4700412549401891 f(x_78) + 0.3217685371860413 f(x_79)
   + 0.36831722012872 f(x_80) + 0.3091450662369439 f(x_81)
   + 0.6366771626949543 f(x_82) + 0.7303426215036607 f(x_83)
   + 0.4433807821018255 f(x_84) + 0.5118451702996591 f(x_85)
   + 0.4948119765317389 f(x_86) + 0.5830732373826251 f(x_87)
   + 0.6050639907934486 f(x_88) + 0.3230979475569062 f(x_89)
   + 0.5858608718817925 f(x_90) + 0.5626235161034694 f(x_91)
   + 0.6090666947375376 f(x_92) + 0.63992645183777 f(x_93)
   + 0.5569063151634023 f(x_94) + 0.2902683461833305 f(x_95)
   + 0.2891310194591686 f(x_96) + 0.7293072540623253 f(x_97)
   + 0.7355392251540269 f(x_98) + 0.6986141465604601 f(x_99)
Subject To
 constr1_x_0: - 0.05 x_0 + self.alpha*x_0+1/(log_base) = 0.5
 constr2_x_0:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_0+1/(log_base)))
   + f(x_0) = 7.272540897341719
 constr1_x_1: - 0.05 x_1 + self.alpha*x_1+1/(log_base)
   = 0.3333333333333333
 constr2_x_1:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_1+1/(log_base)))
   + f(x_1) = 7.860596883514527
 constr1_x_2: - 0.05 x_2 + self.alpha*x_2+1/(log_base) = 0.25
 constr2_x_2:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_2+1/(log_base)))
   + f(x_2) = 7.603568033847865
 constr1_x_3: - 0.05 x_3 + self.alpha*x_3+1/(log_base) = 0.2
 constr2_x_3:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_3+1/(log_base)))
   + f(x_3) = 7.212567439010781
 constr1_x_4: - 0.05 x_4 + self.alpha*x_4+1/(log_base)
   = 0.1666666666666667
 constr2_x_4:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_4+1/(log_base)))
   + f(x_4) = 6.829281696822197
 constr1_x_5: - 0.05 x_5 + self.alpha*x_5+1/(log_base)
   = 0.1428571428571428
 constr2_x_5:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_5+1/(log_base)))
   + f(x_5) = 6.484106534881927
 constr1_x_6: - 0.05 x_6 + self.alpha*x_6+1/(log_base) = 0.125
 constr2_x_6:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_6+1/(log_base)))
   + f(x_6) = 6.180128151318439
 constr1_x_7: - 0.05 x_7 + self.alpha*x_7+1/(log_base)
   = 0.1111111111111111
 constr2_x_7:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_7+1/(log_base)))
   + f(x_7) = 5.913455556386026
 constr1_x_8: - 0.05 x_8 + self.alpha*x_8+1/(log_base) = 0.1
 constr2_x_8:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_8+1/(log_base)))
   + f(x_8) = 5.678873587267566
 constr1_x_9: - 0.05 x_9 + self.alpha*x_9+1/(log_base)
   = 0.0909090909090909
 constr2_x_9:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_9+1/(log_base)))
   + f(x_9) = 5.471462164050473
 constr1_x_10: - 0.05 x_10 + self.alpha*x_10+1/(log_base) = 0.5
 constr2_x_10:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_10+1/(log_base)))
   + f(x_10) = 7.272540897341719
 constr1_x_11: - 0.05 x_11 + self.alpha*x_11+1/(log_base)
   = 0.3333333333333333
 constr2_x_11:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_11+1/(log_base)))
   + f(x_11) = 7.860596883514527
 constr1_x_12: - 0.05 x_12 + self.alpha*x_12+1/(log_base) = 0.25
 constr2_x_12:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_12+1/(log_base)))
   + f(x_12) = 7.603568033847865
 constr1_x_13: - 0.05 x_13 + self.alpha*x_13+1/(log_base) = 0.2
 constr2_x_13:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_13+1/(log_base)))
   + f(x_13) = 7.212567439010781
 constr1_x_14: - 0.05 x_14 + self.alpha*x_14+1/(log_base)
   = 0.1666666666666667
 constr2_x_14:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_14+1/(log_base)))
   + f(x_14) = 6.829281696822197
 constr1_x_15: - 0.05 x_15 + self.alpha*x_15+1/(log_base)
   = 0.1428571428571428
 constr2_x_15:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_15+1/(log_base)))
   + f(x_15) = 6.484106534881927
 constr1_x_16: - 0.05 x_16 + self.alpha*x_16+1/(log_base) = 0.125
 constr2_x_16:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_16+1/(log_base)))
   + f(x_16) = 6.180128151318439
 constr1_x_17: - 0.05 x_17 + self.alpha*x_17+1/(log_base)
   = 0.1111111111111111
 constr2_x_17:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_17+1/(log_base)))
   + f(x_17) = 5.913455556386026
 constr1_x_18: - 0.05 x_18 + self.alpha*x_18+1/(log_base) = 0.1
 constr2_x_18:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_18+1/(log_base)))
   + f(x_18) = 5.678873587267566
 constr1_x_19: - 0.05 x_19 + self.alpha*x_19+1/(log_base)
   = 0.0909090909090909
 constr2_x_19:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_19+1/(log_base)))
   + f(x_19) = 5.471462164050473
 constr1_x_20: - 0.05 x_20 + self.alpha*x_20+1/(log_base) = 0.5
 constr2_x_20:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_20+1/(log_base)))
   + f(x_20) = 7.272540897341719
 constr1_x_21: - 0.05 x_21 + self.alpha*x_21+1/(log_base)
   = 0.3333333333333333
 constr2_x_21:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_21+1/(log_base)))
   + f(x_21) = 7.860596883514527
 constr1_x_22: - 0.05 x_22 + self.alpha*x_22+1/(log_base) = 0.25
 constr2_x_22:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_22+1/(log_base)))
   + f(x_22) = 7.603568033847865
 constr1_x_23: - 0.05 x_23 + self.alpha*x_23+1/(log_base) = 0.2
 constr2_x_23:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_23+1/(log_base)))
   + f(x_23) = 7.212567439010781
 constr1_x_24: - 0.05 x_24 + self.alpha*x_24+1/(log_base)
   = 0.1666666666666667
 constr2_x_24:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_24+1/(log_base)))
   + f(x_24) = 6.829281696822197
 constr1_x_25: - 0.05 x_25 + self.alpha*x_25+1/(log_base)
   = 0.1428571428571428
 constr2_x_25:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_25+1/(log_base)))
   + f(x_25) = 6.484106534881927
 constr1_x_26: - 0.05 x_26 + self.alpha*x_26+1/(log_base) = 0.125
 constr2_x_26:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_26+1/(log_base)))
   + f(x_26) = 6.180128151318439
 constr1_x_27: - 0.05 x_27 + self.alpha*x_27+1/(log_base)
   = 0.1111111111111111
 constr2_x_27:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_27+1/(log_base)))
   + f(x_27) = 5.913455556386026
 constr1_x_28: - 0.05 x_28 + self.alpha*x_28+1/(log_base) = 0.1
 constr2_x_28:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_28+1/(log_base)))
   + f(x_28) = 5.678873587267566
 constr1_x_29: - 0.05 x_29 + self.alpha*x_29+1/(log_base)
   = 0.0909090909090909
 constr2_x_29:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_29+1/(log_base)))
   + f(x_29) = 5.471462164050473
 constr1_x_30: - 0.05 x_30 + self.alpha*x_30+1/(log_base) = 0.5
 constr2_x_30:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_30+1/(log_base)))
   + f(x_30) = 7.272540897341719
 constr1_x_31: - 0.05 x_31 + self.alpha*x_31+1/(log_base)
   = 0.3333333333333333
 constr2_x_31:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_31+1/(log_base)))
   + f(x_31) = 7.860596883514527
 constr1_x_32: - 0.05 x_32 + self.alpha*x_32+1/(log_base) = 0.25
 constr2_x_32:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_32+1/(log_base)))
   + f(x_32) = 7.603568033847865
 constr1_x_33: - 0.05 x_33 + self.alpha*x_33+1/(log_base) = 0.2
 constr2_x_33:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_33+1/(log_base)))
   + f(x_33) = 7.212567439010781
 constr1_x_34: - 0.05 x_34 + self.alpha*x_34+1/(log_base)
   = 0.1666666666666667
 constr2_x_34:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_34+1/(log_base)))
   + f(x_34) = 6.829281696822197
 constr1_x_35: - 0.05 x_35 + self.alpha*x_35+1/(log_base)
   = 0.1428571428571428
 constr2_x_35:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_35+1/(log_base)))
   + f(x_35) = 6.484106534881927
 constr1_x_36: - 0.05 x_36 + self.alpha*x_36+1/(log_base) = 0.125
 constr2_x_36:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_36+1/(log_base)))
   + f(x_36) = 6.180128151318439
 constr1_x_37: - 0.05 x_37 + self.alpha*x_37+1/(log_base)
   = 0.1111111111111111
 constr2_x_37:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_37+1/(log_base)))
   + f(x_37) = 5.913455556386026
 constr1_x_38: - 0.05 x_38 + self.alpha*x_38+1/(log_base) = 0.1
 constr2_x_38:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_38+1/(log_base)))
   + f(x_38) = 5.678873587267566
 constr1_x_39: - 0.05 x_39 + self.alpha*x_39+1/(log_base)
   = 0.0909090909090909
 constr2_x_39:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_39+1/(log_base)))
   + f(x_39) = 5.471462164050473
 constr1_x_40: - 0.05 x_40 + self.alpha*x_40+1/(log_base) = 0.5
 constr2_x_40:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_40+1/(log_base)))
   + f(x_40) = 7.272540897341719
 constr1_x_41: - 0.05 x_41 + self.alpha*x_41+1/(log_base)
   = 0.3333333333333333
 constr2_x_41:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_41+1/(log_base)))
   + f(x_41) = 7.860596883514527
 constr1_x_42: - 0.05 x_42 + self.alpha*x_42+1/(log_base) = 0.25
 constr2_x_42:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_42+1/(log_base)))
   + f(x_42) = 7.603568033847865
 constr1_x_43: - 0.05 x_43 + self.alpha*x_43+1/(log_base) = 0.2
 constr2_x_43:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_43+1/(log_base)))
   + f(x_43) = 7.212567439010781
 constr1_x_44: - 0.05 x_44 + self.alpha*x_44+1/(log_base)
   = 0.1666666666666667
 constr2_x_44:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_44+1/(log_base)))
   + f(x_44) = 6.829281696822197
 constr1_x_45: - 0.05 x_45 + self.alpha*x_45+1/(log_base)
   = 0.1428571428571428
 constr2_x_45:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_45+1/(log_base)))
   + f(x_45) = 6.484106534881927
 constr1_x_46: - 0.05 x_46 + self.alpha*x_46+1/(log_base) = 0.125
 constr2_x_46:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_46+1/(log_base)))
   + f(x_46) = 6.180128151318439
 constr1_x_47: - 0.05 x_47 + self.alpha*x_47+1/(log_base)
   = 0.1111111111111111
 constr2_x_47:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_47+1/(log_base)))
   + f(x_47) = 5.913455556386026
 constr1_x_48: - 0.05 x_48 + self.alpha*x_48+1/(log_base) = 0.1
 constr2_x_48:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_48+1/(log_base)))
   + f(x_48) = 5.678873587267566
 constr1_x_49: - 0.05 x_49 + self.alpha*x_49+1/(log_base)
   = 0.0909090909090909
 constr2_x_49:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_49+1/(log_base)))
   + f(x_49) = 5.471462164050473
 constr1_x_50: - 0.05 x_50 + self.alpha*x_50+1/(log_base) = 0.5
 constr2_x_50:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_50+1/(log_base)))
   + f(x_50) = 7.272540897341719
 constr1_x_51: - 0.05 x_51 + self.alpha*x_51+1/(log_base)
   = 0.3333333333333333
 constr2_x_51:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_51+1/(log_base)))
   + f(x_51) = 7.860596883514527
 constr1_x_52: - 0.05 x_52 + self.alpha*x_52+1/(log_base) = 0.25
 constr2_x_52:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_52+1/(log_base)))
   + f(x_52) = 7.603568033847865
 constr1_x_53: - 0.05 x_53 + self.alpha*x_53+1/(log_base) = 0.2
 constr2_x_53:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_53+1/(log_base)))
   + f(x_53) = 7.212567439010781
 constr1_x_54: - 0.05 x_54 + self.alpha*x_54+1/(log_base)
   = 0.1666666666666667
 constr2_x_54:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_54+1/(log_base)))
   + f(x_54) = 6.829281696822197
 constr1_x_55: - 0.05 x_55 + self.alpha*x_55+1/(log_base)
   = 0.1428571428571428
 constr2_x_55:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_55+1/(log_base)))
   + f(x_55) = 6.484106534881927
 constr1_x_56: - 0.05 x_56 + self.alpha*x_56+1/(log_base) = 0.125
 constr2_x_56:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_56+1/(log_base)))
   + f(x_56) = 6.180128151318439
 constr1_x_57: - 0.05 x_57 + self.alpha*x_57+1/(log_base)
   = 0.1111111111111111
 constr2_x_57:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_57+1/(log_base)))
   + f(x_57) = 5.913455556386026
 constr1_x_58: - 0.05 x_58 + self.alpha*x_58+1/(log_base) = 0.1
 constr2_x_58:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_58+1/(log_base)))
   + f(x_58) = 5.678873587267566
 constr1_x_59: - 0.05 x_59 + self.alpha*x_59+1/(log_base)
   = 0.0909090909090909
 constr2_x_59:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_59+1/(log_base)))
   + f(x_59) = 5.471462164050473
 constr1_x_60: - 0.05 x_60 + self.alpha*x_60+1/(log_base) = 0.5
 constr2_x_60:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_60+1/(log_base)))
   + f(x_60) = 7.272540897341719
 constr1_x_61: - 0.05 x_61 + self.alpha*x_61+1/(log_base)
   = 0.3333333333333333
 constr2_x_61:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_61+1/(log_base)))
   + f(x_61) = 7.860596883514527
 constr1_x_62: - 0.05 x_62 + self.alpha*x_62+1/(log_base) = 0.25
 constr2_x_62:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_62+1/(log_base)))
   + f(x_62) = 7.603568033847865
 constr1_x_63: - 0.05 x_63 + self.alpha*x_63+1/(log_base) = 0.2
 constr2_x_63:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_63+1/(log_base)))
   + f(x_63) = 7.212567439010781
 constr1_x_64: - 0.05 x_64 + self.alpha*x_64+1/(log_base)
   = 0.1666666666666667
 constr2_x_64:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_64+1/(log_base)))
   + f(x_64) = 6.829281696822197
 constr1_x_65: - 0.05 x_65 + self.alpha*x_65+1/(log_base)
   = 0.1428571428571428
 constr2_x_65:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_65+1/(log_base)))
   + f(x_65) = 6.484106534881927
 constr1_x_66: - 0.05 x_66 + self.alpha*x_66+1/(log_base) = 0.125
 constr2_x_66:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_66+1/(log_base)))
   + f(x_66) = 6.180128151318439
 constr1_x_67: - 0.05 x_67 + self.alpha*x_67+1/(log_base)
   = 0.1111111111111111
 constr2_x_67:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_67+1/(log_base)))
   + f(x_67) = 5.913455556386026
 constr1_x_68: - 0.05 x_68 + self.alpha*x_68+1/(log_base) = 0.1
 constr2_x_68:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_68+1/(log_base)))
   + f(x_68) = 5.678873587267566
 constr1_x_69: - 0.05 x_69 + self.alpha*x_69+1/(log_base)
   = 0.0909090909090909
 constr2_x_69:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_69+1/(log_base)))
   + f(x_69) = 5.471462164050473
 constr1_x_70: - 0.05 x_70 + self.alpha*x_70+1/(log_base) = 0.5
 constr2_x_70:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_70+1/(log_base)))
   + f(x_70) = 7.272540897341719
 constr1_x_71: - 0.05 x_71 + self.alpha*x_71+1/(log_base)
   = 0.3333333333333333
 constr2_x_71:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_71+1/(log_base)))
   + f(x_71) = 7.860596883514527
 constr1_x_72: - 0.05 x_72 + self.alpha*x_72+1/(log_base) = 0.25
 constr2_x_72:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_72+1/(log_base)))
   + f(x_72) = 7.603568033847865
 constr1_x_73: - 0.05 x_73 + self.alpha*x_73+1/(log_base) = 0.2
 constr2_x_73:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_73+1/(log_base)))
   + f(x_73) = 7.212567439010781
 constr1_x_74: - 0.05 x_74 + self.alpha*x_74+1/(log_base)
   = 0.1666666666666667
 constr2_x_74:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_74+1/(log_base)))
   + f(x_74) = 6.829281696822197
 constr1_x_75: - 0.05 x_75 + self.alpha*x_75+1/(log_base)
   = 0.1428571428571428
 constr2_x_75:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_75+1/(log_base)))
   + f(x_75) = 6.484106534881927
 constr1_x_76: - 0.05 x_76 + self.alpha*x_76+1/(log_base) = 0.125
 constr2_x_76:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_76+1/(log_base)))
   + f(x_76) = 6.180128151318439
 constr1_x_77: - 0.05 x_77 + self.alpha*x_77+1/(log_base)
   = 0.1111111111111111
 constr2_x_77:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_77+1/(log_base)))
   + f(x_77) = 5.913455556386026
 constr1_x_78: - 0.05 x_78 + self.alpha*x_78+1/(log_base) = 0.1
 constr2_x_78:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_78+1/(log_base)))
   + f(x_78) = 5.678873587267566
 constr1_x_79: - 0.05 x_79 + self.alpha*x_79+1/(log_base)
   = 0.0909090909090909
 constr2_x_79:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_79+1/(log_base)))
   + f(x_79) = 5.471462164050473
 constr1_x_80: - 0.05 x_80 + self.alpha*x_80+1/(log_base) = 0.5
 constr2_x_80:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_80+1/(log_base)))
   + f(x_80) = 7.272540897341719
 constr1_x_81: - 0.05 x_81 + self.alpha*x_81+1/(log_base)
   = 0.3333333333333333
 constr2_x_81:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_81+1/(log_base)))
   + f(x_81) = 7.860596883514527
 constr1_x_82: - 0.05 x_82 + self.alpha*x_82+1/(log_base) = 0.25
 constr2_x_82:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_82+1/(log_base)))
   + f(x_82) = 7.603568033847865
 constr1_x_83: - 0.05 x_83 + self.alpha*x_83+1/(log_base) = 0.2
 constr2_x_83:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_83+1/(log_base)))
   + f(x_83) = 7.212567439010781
 constr1_x_84: - 0.05 x_84 + self.alpha*x_84+1/(log_base)
   = 0.1666666666666667
 constr2_x_84:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_84+1/(log_base)))
   + f(x_84) = 6.829281696822197
 constr1_x_85: - 0.05 x_85 + self.alpha*x_85+1/(log_base)
   = 0.1428571428571428
 constr2_x_85:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_85+1/(log_base)))
   + f(x_85) = 6.484106534881927
 constr1_x_86: - 0.05 x_86 + self.alpha*x_86+1/(log_base) = 0.125
 constr2_x_86:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_86+1/(log_base)))
   + f(x_86) = 6.180128151318439
 constr1_x_87: - 0.05 x_87 + self.alpha*x_87+1/(log_base)
   = 0.1111111111111111
 constr2_x_87:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_87+1/(log_base)))
   + f(x_87) = 5.913455556386026
 constr1_x_88: - 0.05 x_88 + self.alpha*x_88+1/(log_base) = 0.1
 constr2_x_88:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_88+1/(log_base)))
   + f(x_88) = 5.678873587267566
 constr1_x_89: - 0.05 x_89 + self.alpha*x_89+1/(log_base)
   = 0.0909090909090909
 constr2_x_89:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_89+1/(log_base)))
   + f(x_89) = 5.471462164050473
 constr1_x_90: - 0.05 x_90 + self.alpha*x_90+1/(log_base) = 0.5
 constr2_x_90:
   - 7.272540897341719 (np.emath.logn(log_base,self.alpha*x_90+1/(log_base)))
   + f(x_90) = 7.272540897341719
 constr1_x_91: - 0.05 x_91 + self.alpha*x_91+1/(log_base)
   = 0.3333333333333333
 constr2_x_91:
   - 7.860596883514527 (np.emath.logn(log_base,self.alpha*x_91+1/(log_base)))
   + f(x_91) = 7.860596883514527
 constr1_x_92: - 0.05 x_92 + self.alpha*x_92+1/(log_base) = 0.25
 constr2_x_92:
   - 7.603568033847865 (np.emath.logn(log_base,self.alpha*x_92+1/(log_base)))
   + f(x_92) = 7.603568033847865
 constr1_x_93: - 0.05 x_93 + self.alpha*x_93+1/(log_base) = 0.2
 constr2_x_93:
   - 7.212567439010781 (np.emath.logn(log_base,self.alpha*x_93+1/(log_base)))
   + f(x_93) = 7.212567439010781
 constr1_x_94: - 0.05 x_94 + self.alpha*x_94+1/(log_base)
   = 0.1666666666666667
 constr2_x_94:
   - 6.829281696822197 (np.emath.logn(log_base,self.alpha*x_94+1/(log_base)))
   + f(x_94) = 6.829281696822197
 constr1_x_95: - 0.05 x_95 + self.alpha*x_95+1/(log_base)
   = 0.1428571428571428
 constr2_x_95:
   - 6.484106534881927 (np.emath.logn(log_base,self.alpha*x_95+1/(log_base)))
   + f(x_95) = 6.484106534881927
 constr1_x_96: - 0.05 x_96 + self.alpha*x_96+1/(log_base) = 0.125
 constr2_x_96:
   - 6.180128151318439 (np.emath.logn(log_base,self.alpha*x_96+1/(log_base)))
   + f(x_96) = 6.180128151318439
 constr1_x_97: - 0.05 x_97 + self.alpha*x_97+1/(log_base)
   = 0.1111111111111111
 constr2_x_97:
   - 5.913455556386026 (np.emath.logn(log_base,self.alpha*x_97+1/(log_base)))
   + f(x_97) = 5.913455556386026
 constr1_x_98: - 0.05 x_98 + self.alpha*x_98+1/(log_base) = 0.1
 constr2_x_98:
   - 5.678873587267566 (np.emath.logn(log_base,self.alpha*x_98+1/(log_base)))
   + f(x_98) = 5.678873587267566
 constr1_x_99: - 0.05 x_99 + self.alpha*x_99+1/(log_base)
   = 0.0909090909090909
 constr2_x_99:
   - 5.471462164050473 (np.emath.logn(log_base,self.alpha*x_99+1/(log_base)))
   + f(x_99) = 5.471462164050473
 R200: x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_10
   + x_11 + x_12 + x_13 + x_14 + x_15 + x_16 + x_17 + x_18 + x_19 + x_20
   + x_21 + x_22 + x_23 + x_24 + x_25 + x_26 + x_27 + x_28 + x_29 + x_30
   + x_31 + x_32 + x_33 + x_34 + x_35 + x_36 + x_37 + x_38 + x_39 + x_40
   + x_41 + x_42 + x_43 + x_44 + x_45 + x_46 + x_47 + x_48 + x_49 + x_50
   + x_51 + x_52 + x_53 + x_54 + x_55 + x_56 + x_57 + x_58 + x_59 + x_60
   + x_61 + x_62 + x_63 + x_64 + x_65 + x_66 + x_67 + x_68 + x_69 + x_70
   + x_71 + x_72 + x_73 + x_74 + x_75 + x_76 + x_77 + x_78 + x_79 + x_80
   + x_81 + x_82 + x_83 + x_84 + x_85 + x_86 + x_87 + x_88 + x_89 + x_90
   + x_91 + x_92 + x_93 + x_94 + x_95 + x_96 + x_97 + x_98 + x_99 = 5
Bounds
 x_0 <= 5
 0.5 <= self.alpha*x_0+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_0+1/(log_base))) <= -0.4150374992788438
 f(x_0) <= 4.254163709905893
 x_1 <= 5
 0.3333333333333333 <= self.alpha*x_1+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_1+1/(log_base))) <= -0.4906157579814927
 f(x_1) <= 4.004064185322088
 x_2 <= 5
 0.25 <= self.alpha*x_2+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_2+1/(log_base))) <= -0.5
 f(x_2) <= 3.801784016923933
 x_3 <= 5
 0.2 <= self.alpha*x_3+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_3+1/(log_base))) <= -0.4961407271748156
 f(x_3) <= 3.634118985022575
 x_4 <= 5
 0.1666666666666667 <= self.alpha*x_4+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_4+1/(log_base))) <= -0.4886084055306145
 f(x_4) <= 3.492437256018493
 x_5 <= 5
 0.1428571428571428 <= self.alpha*x_5+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_5+1/(log_base))) <= -0.4801399683487006
 f(x_5) <= 3.370827828454115
 x_6 <= 5
 0.125 <= self.alpha*x_6+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_6+1/(log_base))) <= -0.4716791664262813
 f(x_6) <= 3.265090456496964
 x_7 <= 5
 0.1111111111111111 <= self.alpha*x_7+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_7+1/(log_base))) <= -0.463570993835061
 f(x_7) <= 3.172149087112692
 x_8 <= 5
 0.1 <= self.alpha*x_8+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_8+1/(log_base))) <= -0.4559319556497244
 f(x_8) <= 3.089693646737099
 x_9 <= 5
 0.0909090909090909 <= self.alpha*x_9+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_9+1/(log_base))) <= -0.4487850011732098
 f(x_9) <= 3.015952010337909
 x_10 <= 5
 0.5 <= self.alpha*x_10+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_10+1/(log_base))) <= -0.4150374992788438
 f(x_10) <= 4.254163709905893
 x_11 <= 5
 0.3333333333333333 <= self.alpha*x_11+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_11+1/(log_base))) <= -0.4906157579814927
 f(x_11) <= 4.004064185322088
 x_12 <= 5
 0.25 <= self.alpha*x_12+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_12+1/(log_base))) <= -0.5
 f(x_12) <= 3.801784016923933
 x_13 <= 5
 0.2 <= self.alpha*x_13+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_13+1/(log_base))) <= -0.4961407271748156
 f(x_13) <= 3.634118985022575
 x_14 <= 5
 0.1666666666666667 <= self.alpha*x_14+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_14+1/(log_base))) <= -0.4886084055306145
 f(x_14) <= 3.492437256018493
 x_15 <= 5
 0.1428571428571428 <= self.alpha*x_15+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_15+1/(log_base))) <= -0.4801399683487006
 f(x_15) <= 3.370827828454115
 x_16 <= 5
 0.125 <= self.alpha*x_16+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_16+1/(log_base))) <= -0.4716791664262813
 f(x_16) <= 3.265090456496964
 x_17 <= 5
 0.1111111111111111 <= self.alpha*x_17+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_17+1/(log_base))) <= -0.463570993835061
 f(x_17) <= 3.172149087112692
 x_18 <= 5
 0.1 <= self.alpha*x_18+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_18+1/(log_base))) <= -0.4559319556497244
 f(x_18) <= 3.089693646737099
 x_19 <= 5
 0.0909090909090909 <= self.alpha*x_19+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_19+1/(log_base))) <= -0.4487850011732098
 f(x_19) <= 3.015952010337909
 x_20 <= 5
 0.5 <= self.alpha*x_20+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_20+1/(log_base))) <= -0.4150374992788438
 f(x_20) <= 4.254163709905893
 x_21 <= 5
 0.3333333333333333 <= self.alpha*x_21+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_21+1/(log_base))) <= -0.4906157579814927
 f(x_21) <= 4.004064185322088
 x_22 <= 5
 0.25 <= self.alpha*x_22+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_22+1/(log_base))) <= -0.5
 f(x_22) <= 3.801784016923933
 x_23 <= 5
 0.2 <= self.alpha*x_23+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_23+1/(log_base))) <= -0.4961407271748156
 f(x_23) <= 3.634118985022575
 x_24 <= 5
 0.1666666666666667 <= self.alpha*x_24+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_24+1/(log_base))) <= -0.4886084055306145
 f(x_24) <= 3.492437256018493
 x_25 <= 5
 0.1428571428571428 <= self.alpha*x_25+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_25+1/(log_base))) <= -0.4801399683487006
 f(x_25) <= 3.370827828454115
 x_26 <= 5
 0.125 <= self.alpha*x_26+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_26+1/(log_base))) <= -0.4716791664262813
 f(x_26) <= 3.265090456496964
 x_27 <= 5
 0.1111111111111111 <= self.alpha*x_27+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_27+1/(log_base))) <= -0.463570993835061
 f(x_27) <= 3.172149087112692
 x_28 <= 5
 0.1 <= self.alpha*x_28+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_28+1/(log_base))) <= -0.4559319556497244
 f(x_28) <= 3.089693646737099
 x_29 <= 5
 0.0909090909090909 <= self.alpha*x_29+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_29+1/(log_base))) <= -0.4487850011732098
 f(x_29) <= 3.015952010337909
 x_30 <= 5
 0.5 <= self.alpha*x_30+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_30+1/(log_base))) <= -0.4150374992788438
 f(x_30) <= 4.254163709905893
 x_31 <= 5
 0.3333333333333333 <= self.alpha*x_31+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_31+1/(log_base))) <= -0.4906157579814927
 f(x_31) <= 4.004064185322088
 x_32 <= 5
 0.25 <= self.alpha*x_32+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_32+1/(log_base))) <= -0.5
 f(x_32) <= 3.801784016923933
 x_33 <= 5
 0.2 <= self.alpha*x_33+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_33+1/(log_base))) <= -0.4961407271748156
 f(x_33) <= 3.634118985022575
 x_34 <= 5
 0.1666666666666667 <= self.alpha*x_34+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_34+1/(log_base))) <= -0.4886084055306145
 f(x_34) <= 3.492437256018493
 x_35 <= 5
 0.1428571428571428 <= self.alpha*x_35+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_35+1/(log_base))) <= -0.4801399683487006
 f(x_35) <= 3.370827828454115
 x_36 <= 5
 0.125 <= self.alpha*x_36+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_36+1/(log_base))) <= -0.4716791664262813
 f(x_36) <= 3.265090456496964
 x_37 <= 5
 0.1111111111111111 <= self.alpha*x_37+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_37+1/(log_base))) <= -0.463570993835061
 f(x_37) <= 3.172149087112692
 x_38 <= 5
 0.1 <= self.alpha*x_38+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_38+1/(log_base))) <= -0.4559319556497244
 f(x_38) <= 3.089693646737099
 x_39 <= 5
 0.0909090909090909 <= self.alpha*x_39+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_39+1/(log_base))) <= -0.4487850011732098
 f(x_39) <= 3.015952010337909
 x_40 <= 5
 0.5 <= self.alpha*x_40+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_40+1/(log_base))) <= -0.4150374992788438
 f(x_40) <= 4.254163709905893
 x_41 <= 5
 0.3333333333333333 <= self.alpha*x_41+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_41+1/(log_base))) <= -0.4906157579814927
 f(x_41) <= 4.004064185322088
 x_42 <= 5
 0.25 <= self.alpha*x_42+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_42+1/(log_base))) <= -0.5
 f(x_42) <= 3.801784016923933
 x_43 <= 5
 0.2 <= self.alpha*x_43+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_43+1/(log_base))) <= -0.4961407271748156
 f(x_43) <= 3.634118985022575
 x_44 <= 5
 0.1666666666666667 <= self.alpha*x_44+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_44+1/(log_base))) <= -0.4886084055306145
 f(x_44) <= 3.492437256018493
 x_45 <= 5
 0.1428571428571428 <= self.alpha*x_45+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_45+1/(log_base))) <= -0.4801399683487006
 f(x_45) <= 3.370827828454115
 x_46 <= 5
 0.125 <= self.alpha*x_46+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_46+1/(log_base))) <= -0.4716791664262813
 f(x_46) <= 3.265090456496964
 x_47 <= 5
 0.1111111111111111 <= self.alpha*x_47+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_47+1/(log_base))) <= -0.463570993835061
 f(x_47) <= 3.172149087112692
 x_48 <= 5
 0.1 <= self.alpha*x_48+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_48+1/(log_base))) <= -0.4559319556497244
 f(x_48) <= 3.089693646737099
 x_49 <= 5
 0.0909090909090909 <= self.alpha*x_49+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_49+1/(log_base))) <= -0.4487850011732098
 f(x_49) <= 3.015952010337909
 x_50 <= 5
 0.5 <= self.alpha*x_50+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_50+1/(log_base))) <= -0.4150374992788438
 f(x_50) <= 4.254163709905893
 x_51 <= 5
 0.3333333333333333 <= self.alpha*x_51+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_51+1/(log_base))) <= -0.4906157579814927
 f(x_51) <= 4.004064185322088
 x_52 <= 5
 0.25 <= self.alpha*x_52+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_52+1/(log_base))) <= -0.5
 f(x_52) <= 3.801784016923933
 x_53 <= 5
 0.2 <= self.alpha*x_53+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_53+1/(log_base))) <= -0.4961407271748156
 f(x_53) <= 3.634118985022575
 x_54 <= 5
 0.1666666666666667 <= self.alpha*x_54+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_54+1/(log_base))) <= -0.4886084055306145
 f(x_54) <= 3.492437256018493
 x_55 <= 5
 0.1428571428571428 <= self.alpha*x_55+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_55+1/(log_base))) <= -0.4801399683487006
 f(x_55) <= 3.370827828454115
 x_56 <= 5
 0.125 <= self.alpha*x_56+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_56+1/(log_base))) <= -0.4716791664262813
 f(x_56) <= 3.265090456496964
 x_57 <= 5
 0.1111111111111111 <= self.alpha*x_57+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_57+1/(log_base))) <= -0.463570993835061
 f(x_57) <= 3.172149087112692
 x_58 <= 5
 0.1 <= self.alpha*x_58+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_58+1/(log_base))) <= -0.4559319556497244
 f(x_58) <= 3.089693646737099
 x_59 <= 5
 0.0909090909090909 <= self.alpha*x_59+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_59+1/(log_base))) <= -0.4487850011732098
 f(x_59) <= 3.015952010337909
 x_60 <= 5
 0.5 <= self.alpha*x_60+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_60+1/(log_base))) <= -0.4150374992788438
 f(x_60) <= 4.254163709905893
 x_61 <= 5
 0.3333333333333333 <= self.alpha*x_61+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_61+1/(log_base))) <= -0.4906157579814927
 f(x_61) <= 4.004064185322088
 x_62 <= 5
 0.25 <= self.alpha*x_62+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_62+1/(log_base))) <= -0.5
 f(x_62) <= 3.801784016923933
 x_63 <= 5
 0.2 <= self.alpha*x_63+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_63+1/(log_base))) <= -0.4961407271748156
 f(x_63) <= 3.634118985022575
 x_64 <= 5
 0.1666666666666667 <= self.alpha*x_64+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_64+1/(log_base))) <= -0.4886084055306145
 f(x_64) <= 3.492437256018493
 x_65 <= 5
 0.1428571428571428 <= self.alpha*x_65+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_65+1/(log_base))) <= -0.4801399683487006
 f(x_65) <= 3.370827828454115
 x_66 <= 5
 0.125 <= self.alpha*x_66+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_66+1/(log_base))) <= -0.4716791664262813
 f(x_66) <= 3.265090456496964
 x_67 <= 5
 0.1111111111111111 <= self.alpha*x_67+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_67+1/(log_base))) <= -0.463570993835061
 f(x_67) <= 3.172149087112692
 x_68 <= 5
 0.1 <= self.alpha*x_68+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_68+1/(log_base))) <= -0.4559319556497244
 f(x_68) <= 3.089693646737099
 x_69 <= 5
 0.0909090909090909 <= self.alpha*x_69+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_69+1/(log_base))) <= -0.4487850011732098
 f(x_69) <= 3.015952010337909
 x_70 <= 5
 0.5 <= self.alpha*x_70+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_70+1/(log_base))) <= -0.4150374992788438
 f(x_70) <= 4.254163709905893
 x_71 <= 5
 0.3333333333333333 <= self.alpha*x_71+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_71+1/(log_base))) <= -0.4906157579814927
 f(x_71) <= 4.004064185322088
 x_72 <= 5
 0.25 <= self.alpha*x_72+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_72+1/(log_base))) <= -0.5
 f(x_72) <= 3.801784016923933
 x_73 <= 5
 0.2 <= self.alpha*x_73+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_73+1/(log_base))) <= -0.4961407271748156
 f(x_73) <= 3.634118985022575
 x_74 <= 5
 0.1666666666666667 <= self.alpha*x_74+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_74+1/(log_base))) <= -0.4886084055306145
 f(x_74) <= 3.492437256018493
 x_75 <= 5
 0.1428571428571428 <= self.alpha*x_75+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_75+1/(log_base))) <= -0.4801399683487006
 f(x_75) <= 3.370827828454115
 x_76 <= 5
 0.125 <= self.alpha*x_76+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_76+1/(log_base))) <= -0.4716791664262813
 f(x_76) <= 3.265090456496964
 x_77 <= 5
 0.1111111111111111 <= self.alpha*x_77+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_77+1/(log_base))) <= -0.463570993835061
 f(x_77) <= 3.172149087112692
 x_78 <= 5
 0.1 <= self.alpha*x_78+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_78+1/(log_base))) <= -0.4559319556497244
 f(x_78) <= 3.089693646737099
 x_79 <= 5
 0.0909090909090909 <= self.alpha*x_79+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_79+1/(log_base))) <= -0.4487850011732098
 f(x_79) <= 3.015952010337909
 x_80 <= 5
 0.5 <= self.alpha*x_80+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_80+1/(log_base))) <= -0.4150374992788438
 f(x_80) <= 4.254163709905893
 x_81 <= 5
 0.3333333333333333 <= self.alpha*x_81+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_81+1/(log_base))) <= -0.4906157579814927
 f(x_81) <= 4.004064185322088
 x_82 <= 5
 0.25 <= self.alpha*x_82+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_82+1/(log_base))) <= -0.5
 f(x_82) <= 3.801784016923933
 x_83 <= 5
 0.2 <= self.alpha*x_83+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_83+1/(log_base))) <= -0.4961407271748156
 f(x_83) <= 3.634118985022575
 x_84 <= 5
 0.1666666666666667 <= self.alpha*x_84+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_84+1/(log_base))) <= -0.4886084055306145
 f(x_84) <= 3.492437256018493
 x_85 <= 5
 0.1428571428571428 <= self.alpha*x_85+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_85+1/(log_base))) <= -0.4801399683487006
 f(x_85) <= 3.370827828454115
 x_86 <= 5
 0.125 <= self.alpha*x_86+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_86+1/(log_base))) <= -0.4716791664262813
 f(x_86) <= 3.265090456496964
 x_87 <= 5
 0.1111111111111111 <= self.alpha*x_87+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_87+1/(log_base))) <= -0.463570993835061
 f(x_87) <= 3.172149087112692
 x_88 <= 5
 0.1 <= self.alpha*x_88+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_88+1/(log_base))) <= -0.4559319556497244
 f(x_88) <= 3.089693646737099
 x_89 <= 5
 0.0909090909090909 <= self.alpha*x_89+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_89+1/(log_base))) <= -0.4487850011732098
 f(x_89) <= 3.015952010337909
 x_90 <= 5
 0.5 <= self.alpha*x_90+1/(log_base) <= 0.75
 -1 <= (np.emath.logn(log_base,self.alpha*x_90+1/(log_base))) <= -0.4150374992788438
 f(x_90) <= 4.254163709905893
 x_91 <= 5
 0.3333333333333333 <= self.alpha*x_91+1/(log_base) <= 0.5833333333333333
 -1 <= (np.emath.logn(log_base,self.alpha*x_91+1/(log_base))) <= -0.4906157579814927
 f(x_91) <= 4.004064185322088
 x_92 <= 5
 0.25 <= self.alpha*x_92+1/(log_base) <= 0.5
 -1 <= (np.emath.logn(log_base,self.alpha*x_92+1/(log_base))) <= -0.5
 f(x_92) <= 3.801784016923933
 x_93 <= 5
 0.2 <= self.alpha*x_93+1/(log_base) <= 0.45
 -1 <= (np.emath.logn(log_base,self.alpha*x_93+1/(log_base))) <= -0.4961407271748156
 f(x_93) <= 3.634118985022575
 x_94 <= 5
 0.1666666666666667 <= self.alpha*x_94+1/(log_base) <= 0.4166666666666666
 -1 <= (np.emath.logn(log_base,self.alpha*x_94+1/(log_base))) <= -0.4886084055306145
 f(x_94) <= 3.492437256018493
 x_95 <= 5
 0.1428571428571428 <= self.alpha*x_95+1/(log_base) <= 0.3928571428571428
 -1 <= (np.emath.logn(log_base,self.alpha*x_95+1/(log_base))) <= -0.4801399683487006
 f(x_95) <= 3.370827828454115
 x_96 <= 5
 0.125 <= self.alpha*x_96+1/(log_base) <= 0.375
 -1 <= (np.emath.logn(log_base,self.alpha*x_96+1/(log_base))) <= -0.4716791664262813
 f(x_96) <= 3.265090456496964
 x_97 <= 5
 0.1111111111111111 <= self.alpha*x_97+1/(log_base) <= 0.3611111111111111
 -1 <= (np.emath.logn(log_base,self.alpha*x_97+1/(log_base))) <= -0.463570993835061
 f(x_97) <= 3.172149087112692
 x_98 <= 5
 0.1 <= self.alpha*x_98+1/(log_base) <= 0.35
 -1 <= (np.emath.logn(log_base,self.alpha*x_98+1/(log_base))) <= -0.4559319556497244
 f(x_98) <= 3.089693646737099
 x_99 <= 5
 0.0909090909090909 <= self.alpha*x_99+1/(log_base) <= 0.3409090909090909
 -1 <= (np.emath.logn(log_base,self.alpha*x_99+1/(log_base))) <= -0.4487850011732098
 f(x_99) <= 3.015952010337909
Generals
 x_0 x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_10 x_11 x_12 x_13 x_14 x_15 x_16
 x_17 x_18 x_19 x_20 x_21 x_22 x_23 x_24 x_25 x_26 x_27 x_28 x_29 x_30 x_31
 x_32 x_33 x_34 x_35 x_36 x_37 x_38 x_39 x_40 x_41 x_42 x_43 x_44 x_45 x_46
 x_47 x_48 x_49 x_50 x_51 x_52 x_53 x_54 x_55 x_56 x_57 x_58 x_59 x_60 x_61
 x_62 x_63 x_64 x_65 x_66 x_67 x_68 x_69 x_70 x_71 x_72 x_73 x_74 x_75 x_76
 x_77 x_78 x_79 x_80 x_81 x_82 x_83 x_84 x_85 x_86 x_87 x_88 x_89 x_90 x_91
 x_92 x_93 x_94 x_95 x_96 x_97 x_98 x_99
General Constraints
 GC0: (np.emath.logn(log_base,self.alpha*x_0+1/(log_base))) =
   LOG_2 ( self.alpha*x_0+1/(log_base) )
 GC1: (np.emath.logn(log_base,self.alpha*x_1+1/(log_base))) =
   LOG_3 ( self.alpha*x_1+1/(log_base) )
 GC2: (np.emath.logn(log_base,self.alpha*x_2+1/(log_base))) =
   LOG_4 ( self.alpha*x_2+1/(log_base) )
 GC3: (np.emath.logn(log_base,self.alpha*x_3+1/(log_base))) =
   LOG_5 ( self.alpha*x_3+1/(log_base) )
 GC4: (np.emath.logn(log_base,self.alpha*x_4+1/(log_base))) =
   LOG_6 ( self.alpha*x_4+1/(log_base) )
 GC5: (np.emath.logn(log_base,self.alpha*x_5+1/(log_base))) =
   LOG_7 ( self.alpha*x_5+1/(log_base) )
 GC6: (np.emath.logn(log_base,self.alpha*x_6+1/(log_base))) =
   LOG_8 ( self.alpha*x_6+1/(log_base) )
 GC7: (np.emath.logn(log_base,self.alpha*x_7+1/(log_base))) =
   LOG_9 ( self.alpha*x_7+1/(log_base) )
 GC8: (np.emath.logn(log_base,self.alpha*x_8+1/(log_base))) =
   LOG_10 ( self.alpha*x_8+1/(log_base) )
 GC9: (np.emath.logn(log_base,self.alpha*x_9+1/(log_base))) =
   LOG_11 ( self.alpha*x_9+1/(log_base) )
 GC10: (np.emath.logn(log_base,self.alpha*x_10+1/(log_base))) =
   LOG_2 ( self.alpha*x_10+1/(log_base) )
 GC11: (np.emath.logn(log_base,self.alpha*x_11+1/(log_base))) =
   LOG_3 ( self.alpha*x_11+1/(log_base) )
 GC12: (np.emath.logn(log_base,self.alpha*x_12+1/(log_base))) =
   LOG_4 ( self.alpha*x_12+1/(log_base) )
 GC13: (np.emath.logn(log_base,self.alpha*x_13+1/(log_base))) =
   LOG_5 ( self.alpha*x_13+1/(log_base) )
 GC14: (np.emath.logn(log_base,self.alpha*x_14+1/(log_base))) =
   LOG_6 ( self.alpha*x_14+1/(log_base) )
 GC15: (np.emath.logn(log_base,self.alpha*x_15+1/(log_base))) =
   LOG_7 ( self.alpha*x_15+1/(log_base) )
 GC16: (np.emath.logn(log_base,self.alpha*x_16+1/(log_base))) =
   LOG_8 ( self.alpha*x_16+1/(log_base) )
 GC17: (np.emath.logn(log_base,self.alpha*x_17+1/(log_base))) =
   LOG_9 ( self.alpha*x_17+1/(log_base) )
 GC18: (np.emath.logn(log_base,self.alpha*x_18+1/(log_base))) =
   LOG_10 ( self.alpha*x_18+1/(log_base) )
 GC19: (np.emath.logn(log_base,self.alpha*x_19+1/(log_base))) =
   LOG_11 ( self.alpha*x_19+1/(log_base) )
 GC20: (np.emath.logn(log_base,self.alpha*x_20+1/(log_base))) =
   LOG_2 ( self.alpha*x_20+1/(log_base) )
 GC21: (np.emath.logn(log_base,self.alpha*x_21+1/(log_base))) =
   LOG_3 ( self.alpha*x_21+1/(log_base) )
 GC22: (np.emath.logn(log_base,self.alpha*x_22+1/(log_base))) =
   LOG_4 ( self.alpha*x_22+1/(log_base) )
 GC23: (np.emath.logn(log_base,self.alpha*x_23+1/(log_base))) =
   LOG_5 ( self.alpha*x_23+1/(log_base) )
 GC24: (np.emath.logn(log_base,self.alpha*x_24+1/(log_base))) =
   LOG_6 ( self.alpha*x_24+1/(log_base) )
 GC25: (np.emath.logn(log_base,self.alpha*x_25+1/(log_base))) =
   LOG_7 ( self.alpha*x_25+1/(log_base) )
 GC26: (np.emath.logn(log_base,self.alpha*x_26+1/(log_base))) =
   LOG_8 ( self.alpha*x_26+1/(log_base) )
 GC27: (np.emath.logn(log_base,self.alpha*x_27+1/(log_base))) =
   LOG_9 ( self.alpha*x_27+1/(log_base) )
 GC28: (np.emath.logn(log_base,self.alpha*x_28+1/(log_base))) =
   LOG_10 ( self.alpha*x_28+1/(log_base) )
 GC29: (np.emath.logn(log_base,self.alpha*x_29+1/(log_base))) =
   LOG_11 ( self.alpha*x_29+1/(log_base) )
 GC30: (np.emath.logn(log_base,self.alpha*x_30+1/(log_base))) =
   LOG_2 ( self.alpha*x_30+1/(log_base) )
 GC31: (np.emath.logn(log_base,self.alpha*x_31+1/(log_base))) =
   LOG_3 ( self.alpha*x_31+1/(log_base) )
 GC32: (np.emath.logn(log_base,self.alpha*x_32+1/(log_base))) =
   LOG_4 ( self.alpha*x_32+1/(log_base) )
 GC33: (np.emath.logn(log_base,self.alpha*x_33+1/(log_base))) =
   LOG_5 ( self.alpha*x_33+1/(log_base) )
 GC34: (np.emath.logn(log_base,self.alpha*x_34+1/(log_base))) =
   LOG_6 ( self.alpha*x_34+1/(log_base) )
 GC35: (np.emath.logn(log_base,self.alpha*x_35+1/(log_base))) =
   LOG_7 ( self.alpha*x_35+1/(log_base) )
 GC36: (np.emath.logn(log_base,self.alpha*x_36+1/(log_base))) =
   LOG_8 ( self.alpha*x_36+1/(log_base) )
 GC37: (np.emath.logn(log_base,self.alpha*x_37+1/(log_base))) =
   LOG_9 ( self.alpha*x_37+1/(log_base) )
 GC38: (np.emath.logn(log_base,self.alpha*x_38+1/(log_base))) =
   LOG_10 ( self.alpha*x_38+1/(log_base) )
 GC39: (np.emath.logn(log_base,self.alpha*x_39+1/(log_base))) =
   LOG_11 ( self.alpha*x_39+1/(log_base) )
 GC40: (np.emath.logn(log_base,self.alpha*x_40+1/(log_base))) =
   LOG_2 ( self.alpha*x_40+1/(log_base) )
 GC41: (np.emath.logn(log_base,self.alpha*x_41+1/(log_base))) =
   LOG_3 ( self.alpha*x_41+1/(log_base) )
 GC42: (np.emath.logn(log_base,self.alpha*x_42+1/(log_base))) =
   LOG_4 ( self.alpha*x_42+1/(log_base) )
 GC43: (np.emath.logn(log_base,self.alpha*x_43+1/(log_base))) =
   LOG_5 ( self.alpha*x_43+1/(log_base) )
 GC44: (np.emath.logn(log_base,self.alpha*x_44+1/(log_base))) =
   LOG_6 ( self.alpha*x_44+1/(log_base) )
 GC45: (np.emath.logn(log_base,self.alpha*x_45+1/(log_base))) =
   LOG_7 ( self.alpha*x_45+1/(log_base) )
 GC46: (np.emath.logn(log_base,self.alpha*x_46+1/(log_base))) =
   LOG_8 ( self.alpha*x_46+1/(log_base) )
 GC47: (np.emath.logn(log_base,self.alpha*x_47+1/(log_base))) =
   LOG_9 ( self.alpha*x_47+1/(log_base) )
 GC48: (np.emath.logn(log_base,self.alpha*x_48+1/(log_base))) =
   LOG_10 ( self.alpha*x_48+1/(log_base) )
 GC49: (np.emath.logn(log_base,self.alpha*x_49+1/(log_base))) =
   LOG_11 ( self.alpha*x_49+1/(log_base) )
 GC50: (np.emath.logn(log_base,self.alpha*x_50+1/(log_base))) =
   LOG_2 ( self.alpha*x_50+1/(log_base) )
 GC51: (np.emath.logn(log_base,self.alpha*x_51+1/(log_base))) =
   LOG_3 ( self.alpha*x_51+1/(log_base) )
 GC52: (np.emath.logn(log_base,self.alpha*x_52+1/(log_base))) =
   LOG_4 ( self.alpha*x_52+1/(log_base) )
 GC53: (np.emath.logn(log_base,self.alpha*x_53+1/(log_base))) =
   LOG_5 ( self.alpha*x_53+1/(log_base) )
 GC54: (np.emath.logn(log_base,self.alpha*x_54+1/(log_base))) =
   LOG_6 ( self.alpha*x_54+1/(log_base) )
 GC55: (np.emath.logn(log_base,self.alpha*x_55+1/(log_base))) =
   LOG_7 ( self.alpha*x_55+1/(log_base) )
 GC56: (np.emath.logn(log_base,self.alpha*x_56+1/(log_base))) =
   LOG_8 ( self.alpha*x_56+1/(log_base) )
 GC57: (np.emath.logn(log_base,self.alpha*x_57+1/(log_base))) =
   LOG_9 ( self.alpha*x_57+1/(log_base) )
 GC58: (np.emath.logn(log_base,self.alpha*x_58+1/(log_base))) =
   LOG_10 ( self.alpha*x_58+1/(log_base) )
 GC59: (np.emath.logn(log_base,self.alpha*x_59+1/(log_base))) =
   LOG_11 ( self.alpha*x_59+1/(log_base) )
 GC60: (np.emath.logn(log_base,self.alpha*x_60+1/(log_base))) =
   LOG_2 ( self.alpha*x_60+1/(log_base) )
 GC61: (np.emath.logn(log_base,self.alpha*x_61+1/(log_base))) =
   LOG_3 ( self.alpha*x_61+1/(log_base) )
 GC62: (np.emath.logn(log_base,self.alpha*x_62+1/(log_base))) =
   LOG_4 ( self.alpha*x_62+1/(log_base) )
 GC63: (np.emath.logn(log_base,self.alpha*x_63+1/(log_base))) =
   LOG_5 ( self.alpha*x_63+1/(log_base) )
 GC64: (np.emath.logn(log_base,self.alpha*x_64+1/(log_base))) =
   LOG_6 ( self.alpha*x_64+1/(log_base) )
 GC65: (np.emath.logn(log_base,self.alpha*x_65+1/(log_base))) =
   LOG_7 ( self.alpha*x_65+1/(log_base) )
 GC66: (np.emath.logn(log_base,self.alpha*x_66+1/(log_base))) =
   LOG_8 ( self.alpha*x_66+1/(log_base) )
 GC67: (np.emath.logn(log_base,self.alpha*x_67+1/(log_base))) =
   LOG_9 ( self.alpha*x_67+1/(log_base) )
 GC68: (np.emath.logn(log_base,self.alpha*x_68+1/(log_base))) =
   LOG_10 ( self.alpha*x_68+1/(log_base) )
 GC69: (np.emath.logn(log_base,self.alpha*x_69+1/(log_base))) =
   LOG_11 ( self.alpha*x_69+1/(log_base) )
 GC70: (np.emath.logn(log_base,self.alpha*x_70+1/(log_base))) =
   LOG_2 ( self.alpha*x_70+1/(log_base) )
 GC71: (np.emath.logn(log_base,self.alpha*x_71+1/(log_base))) =
   LOG_3 ( self.alpha*x_71+1/(log_base) )
 GC72: (np.emath.logn(log_base,self.alpha*x_72+1/(log_base))) =
   LOG_4 ( self.alpha*x_72+1/(log_base) )
 GC73: (np.emath.logn(log_base,self.alpha*x_73+1/(log_base))) =
   LOG_5 ( self.alpha*x_73+1/(log_base) )
 GC74: (np.emath.logn(log_base,self.alpha*x_74+1/(log_base))) =
   LOG_6 ( self.alpha*x_74+1/(log_base) )
 GC75: (np.emath.logn(log_base,self.alpha*x_75+1/(log_base))) =
   LOG_7 ( self.alpha*x_75+1/(log_base) )
 GC76: (np.emath.logn(log_base,self.alpha*x_76+1/(log_base))) =
   LOG_8 ( self.alpha*x_76+1/(log_base) )
 GC77: (np.emath.logn(log_base,self.alpha*x_77+1/(log_base))) =
   LOG_9 ( self.alpha*x_77+1/(log_base) )
 GC78: (np.emath.logn(log_base,self.alpha*x_78+1/(log_base))) =
   LOG_10 ( self.alpha*x_78+1/(log_base) )
 GC79: (np.emath.logn(log_base,self.alpha*x_79+1/(log_base))) =
   LOG_11 ( self.alpha*x_79+1/(log_base) )
 GC80: (np.emath.logn(log_base,self.alpha*x_80+1/(log_base))) =
   LOG_2 ( self.alpha*x_80+1/(log_base) )
 GC81: (np.emath.logn(log_base,self.alpha*x_81+1/(log_base))) =
   LOG_3 ( self.alpha*x_81+1/(log_base) )
 GC82: (np.emath.logn(log_base,self.alpha*x_82+1/(log_base))) =
   LOG_4 ( self.alpha*x_82+1/(log_base) )
 GC83: (np.emath.logn(log_base,self.alpha*x_83+1/(log_base))) =
   LOG_5 ( self.alpha*x_83+1/(log_base) )
 GC84: (np.emath.logn(log_base,self.alpha*x_84+1/(log_base))) =
   LOG_6 ( self.alpha*x_84+1/(log_base) )
 GC85: (np.emath.logn(log_base,self.alpha*x_85+1/(log_base))) =
   LOG_7 ( self.alpha*x_85+1/(log_base) )
 GC86: (np.emath.logn(log_base,self.alpha*x_86+1/(log_base))) =
   LOG_8 ( self.alpha*x_86+1/(log_base) )
 GC87: (np.emath.logn(log_base,self.alpha*x_87+1/(log_base))) =
   LOG_9 ( self.alpha*x_87+1/(log_base) )
 GC88: (np.emath.logn(log_base,self.alpha*x_88+1/(log_base))) =
   LOG_10 ( self.alpha*x_88+1/(log_base) )
 GC89: (np.emath.logn(log_base,self.alpha*x_89+1/(log_base))) =
   LOG_11 ( self.alpha*x_89+1/(log_base) )
 GC90: (np.emath.logn(log_base,self.alpha*x_90+1/(log_base))) =
   LOG_2 ( self.alpha*x_90+1/(log_base) )
 GC91: (np.emath.logn(log_base,self.alpha*x_91+1/(log_base))) =
   LOG_3 ( self.alpha*x_91+1/(log_base) )
 GC92: (np.emath.logn(log_base,self.alpha*x_92+1/(log_base))) =
   LOG_4 ( self.alpha*x_92+1/(log_base) )
 GC93: (np.emath.logn(log_base,self.alpha*x_93+1/(log_base))) =
   LOG_5 ( self.alpha*x_93+1/(log_base) )
 GC94: (np.emath.logn(log_base,self.alpha*x_94+1/(log_base))) =
   LOG_6 ( self.alpha*x_94+1/(log_base) )
 GC95: (np.emath.logn(log_base,self.alpha*x_95+1/(log_base))) =
   LOG_7 ( self.alpha*x_95+1/(log_base) )
 GC96: (np.emath.logn(log_base,self.alpha*x_96+1/(log_base))) =
   LOG_8 ( self.alpha*x_96+1/(log_base) )
 GC97: (np.emath.logn(log_base,self.alpha*x_97+1/(log_base))) =
   LOG_9 ( self.alpha*x_97+1/(log_base) )
 GC98: (np.emath.logn(log_base,self.alpha*x_98+1/(log_base))) =
   LOG_10 ( self.alpha*x_98+1/(log_base) )
 GC99: (np.emath.logn(log_base,self.alpha*x_99+1/(log_base))) =
   LOG_11 ( self.alpha*x_99+1/(log_base) )
End
