\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.5104744470337095 f(x_0) + 0.4507547334622384 f(x_1)
   + 0.4671249876918867 f(x_2) + 0.502428841493971 f(x_3)
   + 0.4016355942572602 f(x_4) + 0.7221677913287201 f(x_5)
   + 0.6770165297514776 f(x_6) + 0.2552931975163762 f(x_7)
   + 0.3677780298730122 f(x_8) + 0.2707502882477412 f(x_9)
   + 0.6164024084985864 f(x_10) + 0.619119797919684 f(x_11)
   + 0.5391849889148856 f(x_12) + 0.464776911392872 f(x_13)
   + 0.6719245328829271 f(x_14) + 0.484118670029111 f(x_15)
   + 0.5272989226896093 f(x_16) + 0.2559572212632016 f(x_17)
   + 0.4077374565161142 f(x_18) + 0.7308925795862501 f(x_19)
   + 0.3328096738990581 f(x_20) + 0.5986462113060844 f(x_21)
   + 0.4710091719420508 f(x_22) + 0.3467306403395368 f(x_23)
   + 0.3143499258278665 f(x_24) + 0.3456249911696667 f(x_25)
   + 0.3629154882158315 f(x_26) + 0.6833164871542405 f(x_27)
   + 0.5884859671044915 f(x_28) + 0.3781865599979138 f(x_29)
   + 0.4630295278383711 f(x_30) + 0.4009955947463967 f(x_31)
   + 0.4976237464046464 f(x_32) + 0.6429873067846485 f(x_33)
   + 0.412708856295754 f(x_34) + 0.5401405365138712 f(x_35)
   + 0.5884126936156002 f(x_36) + 0.4741970143690526 f(x_37)
   + 0.3297679515452565 f(x_38) + 0.5537435908042343 f(x_39)
   + 0.2749417648709696 f(x_40) + 0.3268877445250214 f(x_41)
   + 0.5043377879912623 f(x_42) + 0.2652588875411629 f(x_43)
   + 0.3854565922631978 f(x_44) + 0.7251041117342264 f(x_45)
   + 0.6926524491634026 f(x_46) + 0.6413590552110295 f(x_47)
   + 0.3113104891226078 f(x_48) + 0.2843944271159142 f(x_49)
   + 0.539942674932506 f(x_50) + 0.4599774262594522 f(x_51)
   + 0.429516505887769 f(x_52) + 0.6326431594625621 f(x_53)
   + 0.6859874324686097 f(x_54) + 0.5362829327996714 f(x_55)
   + 0.2968255860880297 f(x_56) + 0.2679986603312449 f(x_57)
   + 0.2737481583093826 f(x_58) + 0.4463375044045059 f(x_59)
   + 0.6131511054589015 f(x_60) + 0.3096973943999045 f(x_61)
   + 0.2817740313731902 f(x_62) + 0.303247761891868 f(x_63)
   + 0.5575006987403239 f(x_64) + 0.7061159197695321 f(x_65)
   + 0.6982383360928003 f(x_66) + 0.716065145770179 f(x_67)
   + 0.749965906544759 f(x_68) + 0.4416285056272963 f(x_69)
   + 0.6156771690923888 f(x_70) + 0.4901603782667028 f(x_71)
   + 0.3009291982150343 f(x_72) + 0.4295017035668907 f(x_73)
   + 0.3613026441443115 f(x_74) + 0.2598126962745195 f(x_75)
   + 0.4640949574441879 f(x_76) + 0.5117763279645342 f(x_77)
   + 0.2501860787899154 f(x_78) + 0.4756214002888191 f(x_79)
   + 0.5894216898517532 f(x_80) + 0.4875532117528248 f(x_81)
   + 0.4382085322414782 f(x_82) + 0.464551234486029 f(x_83)
   + 0.5069824932225675 f(x_84) + 0.6372515646549708 f(x_85)
   + 0.5496619037817823 f(x_86) + 0.4973784850140387 f(x_87)
   + 0.272664895968479 f(x_88) + 0.7143140217074287 f(x_89)
   + 0.542767927755034 f(x_90) + 0.3232669701848971 f(x_91)
   + 0.7311544266497128 f(x_92) + 0.5946440611917706 f(x_93)
   + 0.4020249669046804 f(x_94) + 0.5821931197386125 f(x_95)
   + 0.5865226980794216 f(x_96) + 0.4703242340135535 f(x_97)
   + 0.3552140567351588 f(x_98) + 0.5531829396564711 f(x_99)
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
