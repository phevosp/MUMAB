\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  77.36126688562925 f(x_0) + 19.17612887860397 f(x_1)
   + 11.3394836105547 f(x_2) + 32.39618647234379 f(x_3)
   + 72.08829744394532 f(x_4) + 49.56246536885494 f(x_5)
   + 60.39306395348326 f(x_6) + 99.20888500759555 f(x_7)
   + 30.97384006687071 f(x_8) + 61.93005507240971 f(x_9)
   + 100.0489855705769 f(x_10) + 12.62327795231556 f(x_11)
   + 22.61238614286788 f(x_12) + 98.74779560985415 f(x_13)
   + 80.96377232011665 f(x_14) + 55.34366151672687 f(x_15)
   + 70.43034122173705 f(x_16) + 53.19608139812703 f(x_17)
   + 33.93666283168991 f(x_18) + 42.82912551385375 f(x_19)
   + 51.81042378953953 f(x_20) + 35.38851062186982 f(x_21)
   + 5.034954803032075 f(x_22) + 66.44250220402832 f(x_23)
   + 43.8180212404081 f(x_24) + 47.62341394493905 f(x_25)
   + 97.03597760679212 f(x_26) + 35.70812853173453 f(x_27)
   + 63.79531586905181 f(x_28) + 97.19430790962902 f(x_29)
   + 55.85275809019073 f(x_30) + 58.6535420032927 f(x_31)
   + 70.84412934763849 f(x_32) + 22.00603813099076 f(x_33)
   + 77.79743056902596 f(x_34) + 36.41378410700774 f(x_35)
   + 12.01655886367658 f(x_36) + 1.197965467169703 f(x_37)
   + 68.67668858355016 f(x_38) + 87.57376149549887 f(x_39)
   + 83.1958783926638 f(x_40) + 55.64232953641688 f(x_41)
   + 58.62213219363245 f(x_42) + 77.52028101172739 f(x_43)
   + 3.150473987071322 f(x_44) + 5.211031454243541 f(x_45)
   + 90.87449985457174 f(x_46) + 20.39352117924064 f(x_47)
   + 8.725734952946951 f(x_48) + 82.32920340188325 f(x_49)
   + 51.80178257456966 f(x_50) + 20.57681146132913 f(x_51)
   + 71.13721291736704 f(x_52) + 25.01128883066145 f(x_53)
   + 55.51668511255537 f(x_54) + 17.95456846214404 f(x_55)
   + 83.76903947691005 f(x_56) + 8.873223111920991 f(x_57)
   + 70.02523042733252 f(x_58) + 40.5725109732122 f(x_59)
   + 30.48228866103378 f(x_60) + 88.9325864158405 f(x_61)
   + 40.31553940534081 f(x_62) + 81.14571170370979 f(x_63)
   + 42.67759807711623 f(x_64) + 5.845388513330018 f(x_65)
   + 11.49088105217207 f(x_66) + 2.741478752256154 f(x_67)
   + 25.81798290629055 f(x_68) + 26.23730741385584 f(x_69)
   + 3.066164623009392 f(x_70) + 87.49416890742326 f(x_71)
   + 50.77671617583375 f(x_72) + 80.79939478318492 f(x_73)
   + 25.51582263948077 f(x_74) + 61.44239319517387 f(x_75)
   + 21.57472840475375 f(x_76) + 24.20578400227503 f(x_77)
   + 60.21874574302453 f(x_78) + 16.01878201316871 f(x_79)
   + 66.95739916522507 f(x_80) + 50.61379871244105 f(x_81)
   + 3.156042254599821 f(x_82) + 15.19855808861256 f(x_83)
   + 58.89058333293292 f(x_84) + 93.30876655800766 f(x_85)
   + 73.1114604030396 f(x_86) + 33.11475289045647 f(x_87)
   + 41.88226575757172 f(x_88) + 30.40300333562405 f(x_89)
   + 58.54660127165487 f(x_90) + 32.07907131144741 f(x_91)
   + 76.53581972630894 f(x_92) + 32.58103994214832 f(x_93)
   + 8.266766497580136 f(x_94) + 57.65646485804163 f(x_95)
   + 21.71286709362525 f(x_96) + 39.44119751109483 f(x_97)
   + 50.64509132804468 f(x_98) + 59.85008229742196 f(x_99)
Subject To
 constr1_x_0: - 0.05 x_0 + self.alpha*x_0+1/(log_base) = 0.01
 constr2_x_0:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_0+1/(log_base)))
   + f(x_0) = 2.570194417876937
 constr1_x_1: - 0.05 x_1 + self.alpha*x_1+1/(log_base) = 0.00990099
 constr2_x_1:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_1+1/(log_base)))
   + f(x_1) = 2.563872841518573
 constr1_x_2: - 0.05 x_2 + self.alpha*x_2+1/(log_base)
   = 0.00980392156862745
 constr2_x_2:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_2+1/(log_base)))
   + f(x_2) = 2.557651679863123
 constr1_x_3: - 0.05 x_3 + self.alpha*x_3+1/(log_base)
   = 0.00970873786407767
 constr2_x_3:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_3+1/(log_base)))
   + f(x_3) = 2.551528352782017
 constr1_x_4: - 0.05 x_4 + self.alpha*x_4+1/(log_base)
   = 0.00961538461538462
 constr2_x_4:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_4+1/(log_base)))
   + f(x_4) = 2.545500370626017
 constr1_x_5: - 0.05 x_5 + self.alpha*x_5+1/(log_base)
   = 0.00952380952380952
 constr2_x_5:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_5+1/(log_base)))
   + f(x_5) = 2.539565330230093
 constr1_x_6: - 0.05 x_6 + self.alpha*x_6+1/(log_base)
   = 0.00943396226415094
 constr2_x_6:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_6+1/(log_base)))
   + f(x_6) = 2.533720911130108
 constr1_x_7: - 0.05 x_7 + self.alpha*x_7+1/(log_base)
   = 0.00934579439252336
 constr2_x_7:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_7+1/(log_base)))
   + f(x_7) = 2.527964871978223
 constr1_x_8: - 0.05 x_8 + self.alpha*x_8+1/(log_base)
   = 0.00925925925925926
 constr2_x_8:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_8+1/(log_base)))
   + f(x_8) = 2.522295047144883
 constr1_x_9: - 0.05 x_9 + self.alpha*x_9+1/(log_base)
   = 0.00917431192660551
 constr2_x_9:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_9+1/(log_base)))
   + f(x_9) = 2.516709343496111
 constr1_x_10: - 0.05 x_10 + self.alpha*x_10+1/(log_base) = 0.01
 constr2_x_10:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_10+1/(log_base)))
   + f(x_10) = 2.570194417876937
 constr1_x_11: - 0.05 x_11 + self.alpha*x_11+1/(log_base) = 0.00990099
 constr2_x_11:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_11+1/(log_base)))
   + f(x_11) = 2.563872841518573
 constr1_x_12: - 0.05 x_12 + self.alpha*x_12+1/(log_base)
   = 0.00980392156862745
 constr2_x_12:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_12+1/(log_base)))
   + f(x_12) = 2.557651679863123
 constr1_x_13: - 0.05 x_13 + self.alpha*x_13+1/(log_base)
   = 0.00970873786407767
 constr2_x_13:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_13+1/(log_base)))
   + f(x_13) = 2.551528352782017
 constr1_x_14: - 0.05 x_14 + self.alpha*x_14+1/(log_base)
   = 0.00961538461538462
 constr2_x_14:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_14+1/(log_base)))
   + f(x_14) = 2.545500370626017
 constr1_x_15: - 0.05 x_15 + self.alpha*x_15+1/(log_base)
   = 0.00952380952380952
 constr2_x_15:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_15+1/(log_base)))
   + f(x_15) = 2.539565330230093
 constr1_x_16: - 0.05 x_16 + self.alpha*x_16+1/(log_base)
   = 0.00943396226415094
 constr2_x_16:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_16+1/(log_base)))
   + f(x_16) = 2.533720911130108
 constr1_x_17: - 0.05 x_17 + self.alpha*x_17+1/(log_base)
   = 0.00934579439252336
 constr2_x_17:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_17+1/(log_base)))
   + f(x_17) = 2.527964871978223
 constr1_x_18: - 0.05 x_18 + self.alpha*x_18+1/(log_base)
   = 0.00925925925925926
 constr2_x_18:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_18+1/(log_base)))
   + f(x_18) = 2.522295047144883
 constr1_x_19: - 0.05 x_19 + self.alpha*x_19+1/(log_base)
   = 0.00917431192660551
 constr2_x_19:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_19+1/(log_base)))
   + f(x_19) = 2.516709343496111
 constr1_x_20: - 0.05 x_20 + self.alpha*x_20+1/(log_base) = 0.01
 constr2_x_20:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_20+1/(log_base)))
   + f(x_20) = 2.570194417876937
 constr1_x_21: - 0.05 x_21 + self.alpha*x_21+1/(log_base) = 0.00990099
 constr2_x_21:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_21+1/(log_base)))
   + f(x_21) = 2.563872841518573
 constr1_x_22: - 0.05 x_22 + self.alpha*x_22+1/(log_base)
   = 0.00980392156862745
 constr2_x_22:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_22+1/(log_base)))
   + f(x_22) = 2.557651679863123
 constr1_x_23: - 0.05 x_23 + self.alpha*x_23+1/(log_base)
   = 0.00970873786407767
 constr2_x_23:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_23+1/(log_base)))
   + f(x_23) = 2.551528352782017
 constr1_x_24: - 0.05 x_24 + self.alpha*x_24+1/(log_base)
   = 0.00961538461538462
 constr2_x_24:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_24+1/(log_base)))
   + f(x_24) = 2.545500370626017
 constr1_x_25: - 0.05 x_25 + self.alpha*x_25+1/(log_base)
   = 0.00952380952380952
 constr2_x_25:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_25+1/(log_base)))
   + f(x_25) = 2.539565330230093
 constr1_x_26: - 0.05 x_26 + self.alpha*x_26+1/(log_base)
   = 0.00943396226415094
 constr2_x_26:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_26+1/(log_base)))
   + f(x_26) = 2.533720911130108
 constr1_x_27: - 0.05 x_27 + self.alpha*x_27+1/(log_base)
   = 0.00934579439252336
 constr2_x_27:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_27+1/(log_base)))
   + f(x_27) = 2.527964871978223
 constr1_x_28: - 0.05 x_28 + self.alpha*x_28+1/(log_base)
   = 0.00925925925925926
 constr2_x_28:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_28+1/(log_base)))
   + f(x_28) = 2.522295047144883
 constr1_x_29: - 0.05 x_29 + self.alpha*x_29+1/(log_base)
   = 0.00917431192660551
 constr2_x_29:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_29+1/(log_base)))
   + f(x_29) = 2.516709343496111
 constr1_x_30: - 0.05 x_30 + self.alpha*x_30+1/(log_base) = 0.01
 constr2_x_30:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_30+1/(log_base)))
   + f(x_30) = 2.570194417876937
 constr1_x_31: - 0.05 x_31 + self.alpha*x_31+1/(log_base) = 0.00990099
 constr2_x_31:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_31+1/(log_base)))
   + f(x_31) = 2.563872841518573
 constr1_x_32: - 0.05 x_32 + self.alpha*x_32+1/(log_base)
   = 0.00980392156862745
 constr2_x_32:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_32+1/(log_base)))
   + f(x_32) = 2.557651679863123
 constr1_x_33: - 0.05 x_33 + self.alpha*x_33+1/(log_base)
   = 0.00970873786407767
 constr2_x_33:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_33+1/(log_base)))
   + f(x_33) = 2.551528352782017
 constr1_x_34: - 0.05 x_34 + self.alpha*x_34+1/(log_base)
   = 0.00961538461538462
 constr2_x_34:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_34+1/(log_base)))
   + f(x_34) = 2.545500370626017
 constr1_x_35: - 0.05 x_35 + self.alpha*x_35+1/(log_base)
   = 0.00952380952380952
 constr2_x_35:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_35+1/(log_base)))
   + f(x_35) = 2.539565330230093
 constr1_x_36: - 0.05 x_36 + self.alpha*x_36+1/(log_base)
   = 0.00943396226415094
 constr2_x_36:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_36+1/(log_base)))
   + f(x_36) = 2.533720911130108
 constr1_x_37: - 0.05 x_37 + self.alpha*x_37+1/(log_base)
   = 0.00934579439252336
 constr2_x_37:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_37+1/(log_base)))
   + f(x_37) = 2.527964871978223
 constr1_x_38: - 0.05 x_38 + self.alpha*x_38+1/(log_base)
   = 0.00925925925925926
 constr2_x_38:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_38+1/(log_base)))
   + f(x_38) = 2.522295047144883
 constr1_x_39: - 0.05 x_39 + self.alpha*x_39+1/(log_base)
   = 0.00917431192660551
 constr2_x_39:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_39+1/(log_base)))
   + f(x_39) = 2.516709343496111
 constr1_x_40: - 0.05 x_40 + self.alpha*x_40+1/(log_base) = 0.01
 constr2_x_40:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_40+1/(log_base)))
   + f(x_40) = 2.570194417876937
 constr1_x_41: - 0.05 x_41 + self.alpha*x_41+1/(log_base) = 0.00990099
 constr2_x_41:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_41+1/(log_base)))
   + f(x_41) = 2.563872841518573
 constr1_x_42: - 0.05 x_42 + self.alpha*x_42+1/(log_base)
   = 0.00980392156862745
 constr2_x_42:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_42+1/(log_base)))
   + f(x_42) = 2.557651679863123
 constr1_x_43: - 0.05 x_43 + self.alpha*x_43+1/(log_base)
   = 0.00970873786407767
 constr2_x_43:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_43+1/(log_base)))
   + f(x_43) = 2.551528352782017
 constr1_x_44: - 0.05 x_44 + self.alpha*x_44+1/(log_base)
   = 0.00961538461538462
 constr2_x_44:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_44+1/(log_base)))
   + f(x_44) = 2.545500370626017
 constr1_x_45: - 0.05 x_45 + self.alpha*x_45+1/(log_base)
   = 0.00952380952380952
 constr2_x_45:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_45+1/(log_base)))
   + f(x_45) = 2.539565330230093
 constr1_x_46: - 0.05 x_46 + self.alpha*x_46+1/(log_base)
   = 0.00943396226415094
 constr2_x_46:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_46+1/(log_base)))
   + f(x_46) = 2.533720911130108
 constr1_x_47: - 0.05 x_47 + self.alpha*x_47+1/(log_base)
   = 0.00934579439252336
 constr2_x_47:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_47+1/(log_base)))
   + f(x_47) = 2.527964871978223
 constr1_x_48: - 0.05 x_48 + self.alpha*x_48+1/(log_base)
   = 0.00925925925925926
 constr2_x_48:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_48+1/(log_base)))
   + f(x_48) = 2.522295047144883
 constr1_x_49: - 0.05 x_49 + self.alpha*x_49+1/(log_base)
   = 0.00917431192660551
 constr2_x_49:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_49+1/(log_base)))
   + f(x_49) = 2.516709343496111
 constr1_x_50: - 0.05 x_50 + self.alpha*x_50+1/(log_base) = 0.01
 constr2_x_50:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_50+1/(log_base)))
   + f(x_50) = 2.570194417876937
 constr1_x_51: - 0.05 x_51 + self.alpha*x_51+1/(log_base) = 0.00990099
 constr2_x_51:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_51+1/(log_base)))
   + f(x_51) = 2.563872841518573
 constr1_x_52: - 0.05 x_52 + self.alpha*x_52+1/(log_base)
   = 0.00980392156862745
 constr2_x_52:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_52+1/(log_base)))
   + f(x_52) = 2.557651679863123
 constr1_x_53: - 0.05 x_53 + self.alpha*x_53+1/(log_base)
   = 0.00970873786407767
 constr2_x_53:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_53+1/(log_base)))
   + f(x_53) = 2.551528352782017
 constr1_x_54: - 0.05 x_54 + self.alpha*x_54+1/(log_base)
   = 0.00961538461538462
 constr2_x_54:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_54+1/(log_base)))
   + f(x_54) = 2.545500370626017
 constr1_x_55: - 0.05 x_55 + self.alpha*x_55+1/(log_base)
   = 0.00952380952380952
 constr2_x_55:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_55+1/(log_base)))
   + f(x_55) = 2.539565330230093
 constr1_x_56: - 0.05 x_56 + self.alpha*x_56+1/(log_base)
   = 0.00943396226415094
 constr2_x_56:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_56+1/(log_base)))
   + f(x_56) = 2.533720911130108
 constr1_x_57: - 0.05 x_57 + self.alpha*x_57+1/(log_base)
   = 0.00934579439252336
 constr2_x_57:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_57+1/(log_base)))
   + f(x_57) = 2.527964871978223
 constr1_x_58: - 0.05 x_58 + self.alpha*x_58+1/(log_base)
   = 0.00925925925925926
 constr2_x_58:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_58+1/(log_base)))
   + f(x_58) = 2.522295047144883
 constr1_x_59: - 0.05 x_59 + self.alpha*x_59+1/(log_base)
   = 0.00917431192660551
 constr2_x_59:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_59+1/(log_base)))
   + f(x_59) = 2.516709343496111
 constr1_x_60: - 0.05 x_60 + self.alpha*x_60+1/(log_base) = 0.01
 constr2_x_60:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_60+1/(log_base)))
   + f(x_60) = 2.570194417876937
 constr1_x_61: - 0.05 x_61 + self.alpha*x_61+1/(log_base) = 0.00990099
 constr2_x_61:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_61+1/(log_base)))
   + f(x_61) = 2.563872841518573
 constr1_x_62: - 0.05 x_62 + self.alpha*x_62+1/(log_base)
   = 0.00980392156862745
 constr2_x_62:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_62+1/(log_base)))
   + f(x_62) = 2.557651679863123
 constr1_x_63: - 0.05 x_63 + self.alpha*x_63+1/(log_base)
   = 0.00970873786407767
 constr2_x_63:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_63+1/(log_base)))
   + f(x_63) = 2.551528352782017
 constr1_x_64: - 0.05 x_64 + self.alpha*x_64+1/(log_base)
   = 0.00961538461538462
 constr2_x_64:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_64+1/(log_base)))
   + f(x_64) = 2.545500370626017
 constr1_x_65: - 0.05 x_65 + self.alpha*x_65+1/(log_base)
   = 0.00952380952380952
 constr2_x_65:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_65+1/(log_base)))
   + f(x_65) = 2.539565330230093
 constr1_x_66: - 0.05 x_66 + self.alpha*x_66+1/(log_base)
   = 0.00943396226415094
 constr2_x_66:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_66+1/(log_base)))
   + f(x_66) = 2.533720911130108
 constr1_x_67: - 0.05 x_67 + self.alpha*x_67+1/(log_base)
   = 0.00934579439252336
 constr2_x_67:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_67+1/(log_base)))
   + f(x_67) = 2.527964871978223
 constr1_x_68: - 0.05 x_68 + self.alpha*x_68+1/(log_base)
   = 0.00925925925925926
 constr2_x_68:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_68+1/(log_base)))
   + f(x_68) = 2.522295047144883
 constr1_x_69: - 0.05 x_69 + self.alpha*x_69+1/(log_base)
   = 0.00917431192660551
 constr2_x_69:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_69+1/(log_base)))
   + f(x_69) = 2.516709343496111
 constr1_x_70: - 0.05 x_70 + self.alpha*x_70+1/(log_base) = 0.01
 constr2_x_70:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_70+1/(log_base)))
   + f(x_70) = 2.570194417876937
 constr1_x_71: - 0.05 x_71 + self.alpha*x_71+1/(log_base) = 0.00990099
 constr2_x_71:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_71+1/(log_base)))
   + f(x_71) = 2.563872841518573
 constr1_x_72: - 0.05 x_72 + self.alpha*x_72+1/(log_base)
   = 0.00980392156862745
 constr2_x_72:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_72+1/(log_base)))
   + f(x_72) = 2.557651679863123
 constr1_x_73: - 0.05 x_73 + self.alpha*x_73+1/(log_base)
   = 0.00970873786407767
 constr2_x_73:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_73+1/(log_base)))
   + f(x_73) = 2.551528352782017
 constr1_x_74: - 0.05 x_74 + self.alpha*x_74+1/(log_base)
   = 0.00961538461538462
 constr2_x_74:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_74+1/(log_base)))
   + f(x_74) = 2.545500370626017
 constr1_x_75: - 0.05 x_75 + self.alpha*x_75+1/(log_base)
   = 0.00952380952380952
 constr2_x_75:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_75+1/(log_base)))
   + f(x_75) = 2.539565330230093
 constr1_x_76: - 0.05 x_76 + self.alpha*x_76+1/(log_base)
   = 0.00943396226415094
 constr2_x_76:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_76+1/(log_base)))
   + f(x_76) = 2.533720911130108
 constr1_x_77: - 0.05 x_77 + self.alpha*x_77+1/(log_base)
   = 0.00934579439252336
 constr2_x_77:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_77+1/(log_base)))
   + f(x_77) = 2.527964871978223
 constr1_x_78: - 0.05 x_78 + self.alpha*x_78+1/(log_base)
   = 0.00925925925925926
 constr2_x_78:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_78+1/(log_base)))
   + f(x_78) = 2.522295047144883
 constr1_x_79: - 0.05 x_79 + self.alpha*x_79+1/(log_base)
   = 0.00917431192660551
 constr2_x_79:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_79+1/(log_base)))
   + f(x_79) = 2.516709343496111
 constr1_x_80: - 0.05 x_80 + self.alpha*x_80+1/(log_base) = 0.01
 constr2_x_80:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_80+1/(log_base)))
   + f(x_80) = 2.570194417876937
 constr1_x_81: - 0.05 x_81 + self.alpha*x_81+1/(log_base) = 0.00990099
 constr2_x_81:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_81+1/(log_base)))
   + f(x_81) = 2.563872841518573
 constr1_x_82: - 0.05 x_82 + self.alpha*x_82+1/(log_base)
   = 0.00980392156862745
 constr2_x_82:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_82+1/(log_base)))
   + f(x_82) = 2.557651679863123
 constr1_x_83: - 0.05 x_83 + self.alpha*x_83+1/(log_base)
   = 0.00970873786407767
 constr2_x_83:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_83+1/(log_base)))
   + f(x_83) = 2.551528352782017
 constr1_x_84: - 0.05 x_84 + self.alpha*x_84+1/(log_base)
   = 0.00961538461538462
 constr2_x_84:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_84+1/(log_base)))
   + f(x_84) = 2.545500370626017
 constr1_x_85: - 0.05 x_85 + self.alpha*x_85+1/(log_base)
   = 0.00952380952380952
 constr2_x_85:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_85+1/(log_base)))
   + f(x_85) = 2.539565330230093
 constr1_x_86: - 0.05 x_86 + self.alpha*x_86+1/(log_base)
   = 0.00943396226415094
 constr2_x_86:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_86+1/(log_base)))
   + f(x_86) = 2.533720911130108
 constr1_x_87: - 0.05 x_87 + self.alpha*x_87+1/(log_base)
   = 0.00934579439252336
 constr2_x_87:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_87+1/(log_base)))
   + f(x_87) = 2.527964871978223
 constr1_x_88: - 0.05 x_88 + self.alpha*x_88+1/(log_base)
   = 0.00925925925925926
 constr2_x_88:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_88+1/(log_base)))
   + f(x_88) = 2.522295047144883
 constr1_x_89: - 0.05 x_89 + self.alpha*x_89+1/(log_base)
   = 0.00917431192660551
 constr2_x_89:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_89+1/(log_base)))
   + f(x_89) = 2.516709343496111
 constr1_x_90: - 0.05 x_90 + self.alpha*x_90+1/(log_base) = 0.01
 constr2_x_90:
   - 2.570194417876937 (np.emath.logn(log_base,self.alpha*x_90+1/(log_base)))
   + f(x_90) = 2.570194417876937
 constr1_x_91: - 0.05 x_91 + self.alpha*x_91+1/(log_base) = 0.00990099
 constr2_x_91:
   - 2.563872841518573 (np.emath.logn(log_base,self.alpha*x_91+1/(log_base)))
   + f(x_91) = 2.563872841518573
 constr1_x_92: - 0.05 x_92 + self.alpha*x_92+1/(log_base)
   = 0.00980392156862745
 constr2_x_92:
   - 2.557651679863123 (np.emath.logn(log_base,self.alpha*x_92+1/(log_base)))
   + f(x_92) = 2.557651679863123
 constr1_x_93: - 0.05 x_93 + self.alpha*x_93+1/(log_base)
   = 0.00970873786407767
 constr2_x_93:
   - 2.551528352782017 (np.emath.logn(log_base,self.alpha*x_93+1/(log_base)))
   + f(x_93) = 2.551528352782017
 constr1_x_94: - 0.05 x_94 + self.alpha*x_94+1/(log_base)
   = 0.00961538461538462
 constr2_x_94:
   - 2.545500370626017 (np.emath.logn(log_base,self.alpha*x_94+1/(log_base)))
   + f(x_94) = 2.545500370626017
 constr1_x_95: - 0.05 x_95 + self.alpha*x_95+1/(log_base)
   = 0.00952380952380952
 constr2_x_95:
   - 2.539565330230093 (np.emath.logn(log_base,self.alpha*x_95+1/(log_base)))
   + f(x_95) = 2.539565330230093
 constr1_x_96: - 0.05 x_96 + self.alpha*x_96+1/(log_base)
   = 0.00943396226415094
 constr2_x_96:
   - 2.533720911130108 (np.emath.logn(log_base,self.alpha*x_96+1/(log_base)))
   + f(x_96) = 2.533720911130108
 constr1_x_97: - 0.05 x_97 + self.alpha*x_97+1/(log_base)
   = 0.00934579439252336
 constr2_x_97:
   - 2.527964871978223 (np.emath.logn(log_base,self.alpha*x_97+1/(log_base)))
   + f(x_97) = 2.527964871978223
 constr1_x_98: - 0.05 x_98 + self.alpha*x_98+1/(log_base)
   = 0.00925925925925926
 constr2_x_98:
   - 2.522295047144883 (np.emath.logn(log_base,self.alpha*x_98+1/(log_base)))
   + f(x_98) = 2.522295047144883
 constr1_x_99: - 0.05 x_99 + self.alpha*x_99+1/(log_base)
   = 0.00917431192660551
 constr2_x_99:
   - 2.516709343496111 (np.emath.logn(log_base,self.alpha*x_99+1/(log_base)))
   + f(x_99) = 2.516709343496111
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
 (np.emath.logn(log_base,self.alpha*x_0+1/(log_base))) free
 x_1 <= 5
 (np.emath.logn(log_base,self.alpha*x_1+1/(log_base))) free
 x_2 <= 5
 (np.emath.logn(log_base,self.alpha*x_2+1/(log_base))) free
 x_3 <= 5
 (np.emath.logn(log_base,self.alpha*x_3+1/(log_base))) free
 x_4 <= 5
 (np.emath.logn(log_base,self.alpha*x_4+1/(log_base))) free
 x_5 <= 5
 (np.emath.logn(log_base,self.alpha*x_5+1/(log_base))) free
 x_6 <= 5
 (np.emath.logn(log_base,self.alpha*x_6+1/(log_base))) free
 x_7 <= 5
 (np.emath.logn(log_base,self.alpha*x_7+1/(log_base))) free
 x_8 <= 5
 (np.emath.logn(log_base,self.alpha*x_8+1/(log_base))) free
 x_9 <= 5
 (np.emath.logn(log_base,self.alpha*x_9+1/(log_base))) free
 x_10 <= 5
 (np.emath.logn(log_base,self.alpha*x_10+1/(log_base))) free
 x_11 <= 5
 (np.emath.logn(log_base,self.alpha*x_11+1/(log_base))) free
 x_12 <= 5
 (np.emath.logn(log_base,self.alpha*x_12+1/(log_base))) free
 x_13 <= 5
 (np.emath.logn(log_base,self.alpha*x_13+1/(log_base))) free
 x_14 <= 5
 (np.emath.logn(log_base,self.alpha*x_14+1/(log_base))) free
 x_15 <= 5
 (np.emath.logn(log_base,self.alpha*x_15+1/(log_base))) free
 x_16 <= 5
 (np.emath.logn(log_base,self.alpha*x_16+1/(log_base))) free
 x_17 <= 5
 (np.emath.logn(log_base,self.alpha*x_17+1/(log_base))) free
 x_18 <= 5
 (np.emath.logn(log_base,self.alpha*x_18+1/(log_base))) free
 x_19 <= 5
 (np.emath.logn(log_base,self.alpha*x_19+1/(log_base))) free
 x_20 <= 5
 (np.emath.logn(log_base,self.alpha*x_20+1/(log_base))) free
 x_21 <= 5
 (np.emath.logn(log_base,self.alpha*x_21+1/(log_base))) free
 x_22 <= 5
 (np.emath.logn(log_base,self.alpha*x_22+1/(log_base))) free
 x_23 <= 5
 (np.emath.logn(log_base,self.alpha*x_23+1/(log_base))) free
 x_24 <= 5
 (np.emath.logn(log_base,self.alpha*x_24+1/(log_base))) free
 x_25 <= 5
 (np.emath.logn(log_base,self.alpha*x_25+1/(log_base))) free
 x_26 <= 5
 (np.emath.logn(log_base,self.alpha*x_26+1/(log_base))) free
 x_27 <= 5
 (np.emath.logn(log_base,self.alpha*x_27+1/(log_base))) free
 x_28 <= 5
 (np.emath.logn(log_base,self.alpha*x_28+1/(log_base))) free
 x_29 <= 5
 (np.emath.logn(log_base,self.alpha*x_29+1/(log_base))) free
 x_30 <= 5
 (np.emath.logn(log_base,self.alpha*x_30+1/(log_base))) free
 x_31 <= 5
 (np.emath.logn(log_base,self.alpha*x_31+1/(log_base))) free
 x_32 <= 5
 (np.emath.logn(log_base,self.alpha*x_32+1/(log_base))) free
 x_33 <= 5
 (np.emath.logn(log_base,self.alpha*x_33+1/(log_base))) free
 x_34 <= 5
 (np.emath.logn(log_base,self.alpha*x_34+1/(log_base))) free
 x_35 <= 5
 (np.emath.logn(log_base,self.alpha*x_35+1/(log_base))) free
 x_36 <= 5
 (np.emath.logn(log_base,self.alpha*x_36+1/(log_base))) free
 x_37 <= 5
 (np.emath.logn(log_base,self.alpha*x_37+1/(log_base))) free
 x_38 <= 5
 (np.emath.logn(log_base,self.alpha*x_38+1/(log_base))) free
 x_39 <= 5
 (np.emath.logn(log_base,self.alpha*x_39+1/(log_base))) free
 x_40 <= 5
 (np.emath.logn(log_base,self.alpha*x_40+1/(log_base))) free
 x_41 <= 5
 (np.emath.logn(log_base,self.alpha*x_41+1/(log_base))) free
 x_42 <= 5
 (np.emath.logn(log_base,self.alpha*x_42+1/(log_base))) free
 x_43 <= 5
 (np.emath.logn(log_base,self.alpha*x_43+1/(log_base))) free
 x_44 <= 5
 (np.emath.logn(log_base,self.alpha*x_44+1/(log_base))) free
 x_45 <= 5
 (np.emath.logn(log_base,self.alpha*x_45+1/(log_base))) free
 x_46 <= 5
 (np.emath.logn(log_base,self.alpha*x_46+1/(log_base))) free
 x_47 <= 5
 (np.emath.logn(log_base,self.alpha*x_47+1/(log_base))) free
 x_48 <= 5
 (np.emath.logn(log_base,self.alpha*x_48+1/(log_base))) free
 x_49 <= 5
 (np.emath.logn(log_base,self.alpha*x_49+1/(log_base))) free
 x_50 <= 5
 (np.emath.logn(log_base,self.alpha*x_50+1/(log_base))) free
 x_51 <= 5
 (np.emath.logn(log_base,self.alpha*x_51+1/(log_base))) free
 x_52 <= 5
 (np.emath.logn(log_base,self.alpha*x_52+1/(log_base))) free
 x_53 <= 5
 (np.emath.logn(log_base,self.alpha*x_53+1/(log_base))) free
 x_54 <= 5
 (np.emath.logn(log_base,self.alpha*x_54+1/(log_base))) free
 x_55 <= 5
 (np.emath.logn(log_base,self.alpha*x_55+1/(log_base))) free
 x_56 <= 5
 (np.emath.logn(log_base,self.alpha*x_56+1/(log_base))) free
 x_57 <= 5
 (np.emath.logn(log_base,self.alpha*x_57+1/(log_base))) free
 x_58 <= 5
 (np.emath.logn(log_base,self.alpha*x_58+1/(log_base))) free
 x_59 <= 5
 (np.emath.logn(log_base,self.alpha*x_59+1/(log_base))) free
 x_60 <= 5
 (np.emath.logn(log_base,self.alpha*x_60+1/(log_base))) free
 x_61 <= 5
 (np.emath.logn(log_base,self.alpha*x_61+1/(log_base))) free
 x_62 <= 5
 (np.emath.logn(log_base,self.alpha*x_62+1/(log_base))) free
 x_63 <= 5
 (np.emath.logn(log_base,self.alpha*x_63+1/(log_base))) free
 x_64 <= 5
 (np.emath.logn(log_base,self.alpha*x_64+1/(log_base))) free
 x_65 <= 5
 (np.emath.logn(log_base,self.alpha*x_65+1/(log_base))) free
 x_66 <= 5
 (np.emath.logn(log_base,self.alpha*x_66+1/(log_base))) free
 x_67 <= 5
 (np.emath.logn(log_base,self.alpha*x_67+1/(log_base))) free
 x_68 <= 5
 (np.emath.logn(log_base,self.alpha*x_68+1/(log_base))) free
 x_69 <= 5
 (np.emath.logn(log_base,self.alpha*x_69+1/(log_base))) free
 x_70 <= 5
 (np.emath.logn(log_base,self.alpha*x_70+1/(log_base))) free
 x_71 <= 5
 (np.emath.logn(log_base,self.alpha*x_71+1/(log_base))) free
 x_72 <= 5
 (np.emath.logn(log_base,self.alpha*x_72+1/(log_base))) free
 x_73 <= 5
 (np.emath.logn(log_base,self.alpha*x_73+1/(log_base))) free
 x_74 <= 5
 (np.emath.logn(log_base,self.alpha*x_74+1/(log_base))) free
 x_75 <= 5
 (np.emath.logn(log_base,self.alpha*x_75+1/(log_base))) free
 x_76 <= 5
 (np.emath.logn(log_base,self.alpha*x_76+1/(log_base))) free
 x_77 <= 5
 (np.emath.logn(log_base,self.alpha*x_77+1/(log_base))) free
 x_78 <= 5
 (np.emath.logn(log_base,self.alpha*x_78+1/(log_base))) free
 x_79 <= 5
 (np.emath.logn(log_base,self.alpha*x_79+1/(log_base))) free
 x_80 <= 5
 (np.emath.logn(log_base,self.alpha*x_80+1/(log_base))) free
 x_81 <= 5
 (np.emath.logn(log_base,self.alpha*x_81+1/(log_base))) free
 x_82 <= 5
 (np.emath.logn(log_base,self.alpha*x_82+1/(log_base))) free
 x_83 <= 5
 (np.emath.logn(log_base,self.alpha*x_83+1/(log_base))) free
 x_84 <= 5
 (np.emath.logn(log_base,self.alpha*x_84+1/(log_base))) free
 x_85 <= 5
 (np.emath.logn(log_base,self.alpha*x_85+1/(log_base))) free
 x_86 <= 5
 (np.emath.logn(log_base,self.alpha*x_86+1/(log_base))) free
 x_87 <= 5
 (np.emath.logn(log_base,self.alpha*x_87+1/(log_base))) free
 x_88 <= 5
 (np.emath.logn(log_base,self.alpha*x_88+1/(log_base))) free
 x_89 <= 5
 (np.emath.logn(log_base,self.alpha*x_89+1/(log_base))) free
 x_90 <= 5
 (np.emath.logn(log_base,self.alpha*x_90+1/(log_base))) free
 x_91 <= 5
 (np.emath.logn(log_base,self.alpha*x_91+1/(log_base))) free
 x_92 <= 5
 (np.emath.logn(log_base,self.alpha*x_92+1/(log_base))) free
 x_93 <= 5
 (np.emath.logn(log_base,self.alpha*x_93+1/(log_base))) free
 x_94 <= 5
 (np.emath.logn(log_base,self.alpha*x_94+1/(log_base))) free
 x_95 <= 5
 (np.emath.logn(log_base,self.alpha*x_95+1/(log_base))) free
 x_96 <= 5
 (np.emath.logn(log_base,self.alpha*x_96+1/(log_base))) free
 x_97 <= 5
 (np.emath.logn(log_base,self.alpha*x_97+1/(log_base))) free
 x_98 <= 5
 (np.emath.logn(log_base,self.alpha*x_98+1/(log_base))) free
 x_99 <= 5
 (np.emath.logn(log_base,self.alpha*x_99+1/(log_base))) free
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
   LOG_100 ( self.alpha*x_0+1/(log_base) )
 GC1: (np.emath.logn(log_base,self.alpha*x_1+1/(log_base))) =
   LOG_101 ( self.alpha*x_1+1/(log_base) )
 GC2: (np.emath.logn(log_base,self.alpha*x_2+1/(log_base))) =
   LOG_102 ( self.alpha*x_2+1/(log_base) )
 GC3: (np.emath.logn(log_base,self.alpha*x_3+1/(log_base))) =
   LOG_103 ( self.alpha*x_3+1/(log_base) )
 GC4: (np.emath.logn(log_base,self.alpha*x_4+1/(log_base))) =
   LOG_104 ( self.alpha*x_4+1/(log_base) )
 GC5: (np.emath.logn(log_base,self.alpha*x_5+1/(log_base))) =
   LOG_105 ( self.alpha*x_5+1/(log_base) )
 GC6: (np.emath.logn(log_base,self.alpha*x_6+1/(log_base))) =
   LOG_106 ( self.alpha*x_6+1/(log_base) )
 GC7: (np.emath.logn(log_base,self.alpha*x_7+1/(log_base))) =
   LOG_107 ( self.alpha*x_7+1/(log_base) )
 GC8: (np.emath.logn(log_base,self.alpha*x_8+1/(log_base))) =
   LOG_108 ( self.alpha*x_8+1/(log_base) )
 GC9: (np.emath.logn(log_base,self.alpha*x_9+1/(log_base))) =
   LOG_109 ( self.alpha*x_9+1/(log_base) )
 GC10: (np.emath.logn(log_base,self.alpha*x_10+1/(log_base))) =
   LOG_100 ( self.alpha*x_10+1/(log_base) )
 GC11: (np.emath.logn(log_base,self.alpha*x_11+1/(log_base))) =
   LOG_101 ( self.alpha*x_11+1/(log_base) )
 GC12: (np.emath.logn(log_base,self.alpha*x_12+1/(log_base))) =
   LOG_102 ( self.alpha*x_12+1/(log_base) )
 GC13: (np.emath.logn(log_base,self.alpha*x_13+1/(log_base))) =
   LOG_103 ( self.alpha*x_13+1/(log_base) )
 GC14: (np.emath.logn(log_base,self.alpha*x_14+1/(log_base))) =
   LOG_104 ( self.alpha*x_14+1/(log_base) )
 GC15: (np.emath.logn(log_base,self.alpha*x_15+1/(log_base))) =
   LOG_105 ( self.alpha*x_15+1/(log_base) )
 GC16: (np.emath.logn(log_base,self.alpha*x_16+1/(log_base))) =
   LOG_106 ( self.alpha*x_16+1/(log_base) )
 GC17: (np.emath.logn(log_base,self.alpha*x_17+1/(log_base))) =
   LOG_107 ( self.alpha*x_17+1/(log_base) )
 GC18: (np.emath.logn(log_base,self.alpha*x_18+1/(log_base))) =
   LOG_108 ( self.alpha*x_18+1/(log_base) )
 GC19: (np.emath.logn(log_base,self.alpha*x_19+1/(log_base))) =
   LOG_109 ( self.alpha*x_19+1/(log_base) )
 GC20: (np.emath.logn(log_base,self.alpha*x_20+1/(log_base))) =
   LOG_100 ( self.alpha*x_20+1/(log_base) )
 GC21: (np.emath.logn(log_base,self.alpha*x_21+1/(log_base))) =
   LOG_101 ( self.alpha*x_21+1/(log_base) )
 GC22: (np.emath.logn(log_base,self.alpha*x_22+1/(log_base))) =
   LOG_102 ( self.alpha*x_22+1/(log_base) )
 GC23: (np.emath.logn(log_base,self.alpha*x_23+1/(log_base))) =
   LOG_103 ( self.alpha*x_23+1/(log_base) )
 GC24: (np.emath.logn(log_base,self.alpha*x_24+1/(log_base))) =
   LOG_104 ( self.alpha*x_24+1/(log_base) )
 GC25: (np.emath.logn(log_base,self.alpha*x_25+1/(log_base))) =
   LOG_105 ( self.alpha*x_25+1/(log_base) )
 GC26: (np.emath.logn(log_base,self.alpha*x_26+1/(log_base))) =
   LOG_106 ( self.alpha*x_26+1/(log_base) )
 GC27: (np.emath.logn(log_base,self.alpha*x_27+1/(log_base))) =
   LOG_107 ( self.alpha*x_27+1/(log_base) )
 GC28: (np.emath.logn(log_base,self.alpha*x_28+1/(log_base))) =
   LOG_108 ( self.alpha*x_28+1/(log_base) )
 GC29: (np.emath.logn(log_base,self.alpha*x_29+1/(log_base))) =
   LOG_109 ( self.alpha*x_29+1/(log_base) )
 GC30: (np.emath.logn(log_base,self.alpha*x_30+1/(log_base))) =
   LOG_100 ( self.alpha*x_30+1/(log_base) )
 GC31: (np.emath.logn(log_base,self.alpha*x_31+1/(log_base))) =
   LOG_101 ( self.alpha*x_31+1/(log_base) )
 GC32: (np.emath.logn(log_base,self.alpha*x_32+1/(log_base))) =
   LOG_102 ( self.alpha*x_32+1/(log_base) )
 GC33: (np.emath.logn(log_base,self.alpha*x_33+1/(log_base))) =
   LOG_103 ( self.alpha*x_33+1/(log_base) )
 GC34: (np.emath.logn(log_base,self.alpha*x_34+1/(log_base))) =
   LOG_104 ( self.alpha*x_34+1/(log_base) )
 GC35: (np.emath.logn(log_base,self.alpha*x_35+1/(log_base))) =
   LOG_105 ( self.alpha*x_35+1/(log_base) )
 GC36: (np.emath.logn(log_base,self.alpha*x_36+1/(log_base))) =
   LOG_106 ( self.alpha*x_36+1/(log_base) )
 GC37: (np.emath.logn(log_base,self.alpha*x_37+1/(log_base))) =
   LOG_107 ( self.alpha*x_37+1/(log_base) )
 GC38: (np.emath.logn(log_base,self.alpha*x_38+1/(log_base))) =
   LOG_108 ( self.alpha*x_38+1/(log_base) )
 GC39: (np.emath.logn(log_base,self.alpha*x_39+1/(log_base))) =
   LOG_109 ( self.alpha*x_39+1/(log_base) )
 GC40: (np.emath.logn(log_base,self.alpha*x_40+1/(log_base))) =
   LOG_100 ( self.alpha*x_40+1/(log_base) )
 GC41: (np.emath.logn(log_base,self.alpha*x_41+1/(log_base))) =
   LOG_101 ( self.alpha*x_41+1/(log_base) )
 GC42: (np.emath.logn(log_base,self.alpha*x_42+1/(log_base))) =
   LOG_102 ( self.alpha*x_42+1/(log_base) )
 GC43: (np.emath.logn(log_base,self.alpha*x_43+1/(log_base))) =
   LOG_103 ( self.alpha*x_43+1/(log_base) )
 GC44: (np.emath.logn(log_base,self.alpha*x_44+1/(log_base))) =
   LOG_104 ( self.alpha*x_44+1/(log_base) )
 GC45: (np.emath.logn(log_base,self.alpha*x_45+1/(log_base))) =
   LOG_105 ( self.alpha*x_45+1/(log_base) )
 GC46: (np.emath.logn(log_base,self.alpha*x_46+1/(log_base))) =
   LOG_106 ( self.alpha*x_46+1/(log_base) )
 GC47: (np.emath.logn(log_base,self.alpha*x_47+1/(log_base))) =
   LOG_107 ( self.alpha*x_47+1/(log_base) )
 GC48: (np.emath.logn(log_base,self.alpha*x_48+1/(log_base))) =
   LOG_108 ( self.alpha*x_48+1/(log_base) )
 GC49: (np.emath.logn(log_base,self.alpha*x_49+1/(log_base))) =
   LOG_109 ( self.alpha*x_49+1/(log_base) )
 GC50: (np.emath.logn(log_base,self.alpha*x_50+1/(log_base))) =
   LOG_100 ( self.alpha*x_50+1/(log_base) )
 GC51: (np.emath.logn(log_base,self.alpha*x_51+1/(log_base))) =
   LOG_101 ( self.alpha*x_51+1/(log_base) )
 GC52: (np.emath.logn(log_base,self.alpha*x_52+1/(log_base))) =
   LOG_102 ( self.alpha*x_52+1/(log_base) )
 GC53: (np.emath.logn(log_base,self.alpha*x_53+1/(log_base))) =
   LOG_103 ( self.alpha*x_53+1/(log_base) )
 GC54: (np.emath.logn(log_base,self.alpha*x_54+1/(log_base))) =
   LOG_104 ( self.alpha*x_54+1/(log_base) )
 GC55: (np.emath.logn(log_base,self.alpha*x_55+1/(log_base))) =
   LOG_105 ( self.alpha*x_55+1/(log_base) )
 GC56: (np.emath.logn(log_base,self.alpha*x_56+1/(log_base))) =
   LOG_106 ( self.alpha*x_56+1/(log_base) )
 GC57: (np.emath.logn(log_base,self.alpha*x_57+1/(log_base))) =
   LOG_107 ( self.alpha*x_57+1/(log_base) )
 GC58: (np.emath.logn(log_base,self.alpha*x_58+1/(log_base))) =
   LOG_108 ( self.alpha*x_58+1/(log_base) )
 GC59: (np.emath.logn(log_base,self.alpha*x_59+1/(log_base))) =
   LOG_109 ( self.alpha*x_59+1/(log_base) )
 GC60: (np.emath.logn(log_base,self.alpha*x_60+1/(log_base))) =
   LOG_100 ( self.alpha*x_60+1/(log_base) )
 GC61: (np.emath.logn(log_base,self.alpha*x_61+1/(log_base))) =
   LOG_101 ( self.alpha*x_61+1/(log_base) )
 GC62: (np.emath.logn(log_base,self.alpha*x_62+1/(log_base))) =
   LOG_102 ( self.alpha*x_62+1/(log_base) )
 GC63: (np.emath.logn(log_base,self.alpha*x_63+1/(log_base))) =
   LOG_103 ( self.alpha*x_63+1/(log_base) )
 GC64: (np.emath.logn(log_base,self.alpha*x_64+1/(log_base))) =
   LOG_104 ( self.alpha*x_64+1/(log_base) )
 GC65: (np.emath.logn(log_base,self.alpha*x_65+1/(log_base))) =
   LOG_105 ( self.alpha*x_65+1/(log_base) )
 GC66: (np.emath.logn(log_base,self.alpha*x_66+1/(log_base))) =
   LOG_106 ( self.alpha*x_66+1/(log_base) )
 GC67: (np.emath.logn(log_base,self.alpha*x_67+1/(log_base))) =
   LOG_107 ( self.alpha*x_67+1/(log_base) )
 GC68: (np.emath.logn(log_base,self.alpha*x_68+1/(log_base))) =
   LOG_108 ( self.alpha*x_68+1/(log_base) )
 GC69: (np.emath.logn(log_base,self.alpha*x_69+1/(log_base))) =
   LOG_109 ( self.alpha*x_69+1/(log_base) )
 GC70: (np.emath.logn(log_base,self.alpha*x_70+1/(log_base))) =
   LOG_100 ( self.alpha*x_70+1/(log_base) )
 GC71: (np.emath.logn(log_base,self.alpha*x_71+1/(log_base))) =
   LOG_101 ( self.alpha*x_71+1/(log_base) )
 GC72: (np.emath.logn(log_base,self.alpha*x_72+1/(log_base))) =
   LOG_102 ( self.alpha*x_72+1/(log_base) )
 GC73: (np.emath.logn(log_base,self.alpha*x_73+1/(log_base))) =
   LOG_103 ( self.alpha*x_73+1/(log_base) )
 GC74: (np.emath.logn(log_base,self.alpha*x_74+1/(log_base))) =
   LOG_104 ( self.alpha*x_74+1/(log_base) )
 GC75: (np.emath.logn(log_base,self.alpha*x_75+1/(log_base))) =
   LOG_105 ( self.alpha*x_75+1/(log_base) )
 GC76: (np.emath.logn(log_base,self.alpha*x_76+1/(log_base))) =
   LOG_106 ( self.alpha*x_76+1/(log_base) )
 GC77: (np.emath.logn(log_base,self.alpha*x_77+1/(log_base))) =
   LOG_107 ( self.alpha*x_77+1/(log_base) )
 GC78: (np.emath.logn(log_base,self.alpha*x_78+1/(log_base))) =
   LOG_108 ( self.alpha*x_78+1/(log_base) )
 GC79: (np.emath.logn(log_base,self.alpha*x_79+1/(log_base))) =
   LOG_109 ( self.alpha*x_79+1/(log_base) )
 GC80: (np.emath.logn(log_base,self.alpha*x_80+1/(log_base))) =
   LOG_100 ( self.alpha*x_80+1/(log_base) )
 GC81: (np.emath.logn(log_base,self.alpha*x_81+1/(log_base))) =
   LOG_101 ( self.alpha*x_81+1/(log_base) )
 GC82: (np.emath.logn(log_base,self.alpha*x_82+1/(log_base))) =
   LOG_102 ( self.alpha*x_82+1/(log_base) )
 GC83: (np.emath.logn(log_base,self.alpha*x_83+1/(log_base))) =
   LOG_103 ( self.alpha*x_83+1/(log_base) )
 GC84: (np.emath.logn(log_base,self.alpha*x_84+1/(log_base))) =
   LOG_104 ( self.alpha*x_84+1/(log_base) )
 GC85: (np.emath.logn(log_base,self.alpha*x_85+1/(log_base))) =
   LOG_105 ( self.alpha*x_85+1/(log_base) )
 GC86: (np.emath.logn(log_base,self.alpha*x_86+1/(log_base))) =
   LOG_106 ( self.alpha*x_86+1/(log_base) )
 GC87: (np.emath.logn(log_base,self.alpha*x_87+1/(log_base))) =
   LOG_107 ( self.alpha*x_87+1/(log_base) )
 GC88: (np.emath.logn(log_base,self.alpha*x_88+1/(log_base))) =
   LOG_108 ( self.alpha*x_88+1/(log_base) )
 GC89: (np.emath.logn(log_base,self.alpha*x_89+1/(log_base))) =
   LOG_109 ( self.alpha*x_89+1/(log_base) )
 GC90: (np.emath.logn(log_base,self.alpha*x_90+1/(log_base))) =
   LOG_100 ( self.alpha*x_90+1/(log_base) )
 GC91: (np.emath.logn(log_base,self.alpha*x_91+1/(log_base))) =
   LOG_101 ( self.alpha*x_91+1/(log_base) )
 GC92: (np.emath.logn(log_base,self.alpha*x_92+1/(log_base))) =
   LOG_102 ( self.alpha*x_92+1/(log_base) )
 GC93: (np.emath.logn(log_base,self.alpha*x_93+1/(log_base))) =
   LOG_103 ( self.alpha*x_93+1/(log_base) )
 GC94: (np.emath.logn(log_base,self.alpha*x_94+1/(log_base))) =
   LOG_104 ( self.alpha*x_94+1/(log_base) )
 GC95: (np.emath.logn(log_base,self.alpha*x_95+1/(log_base))) =
   LOG_105 ( self.alpha*x_95+1/(log_base) )
 GC96: (np.emath.logn(log_base,self.alpha*x_96+1/(log_base))) =
   LOG_106 ( self.alpha*x_96+1/(log_base) )
 GC97: (np.emath.logn(log_base,self.alpha*x_97+1/(log_base))) =
   LOG_107 ( self.alpha*x_97+1/(log_base) )
 GC98: (np.emath.logn(log_base,self.alpha*x_98+1/(log_base))) =
   LOG_108 ( self.alpha*x_98+1/(log_base) )
 GC99: (np.emath.logn(log_base,self.alpha*x_99+1/(log_base))) =
   LOG_109 ( self.alpha*x_99+1/(log_base) )
End
