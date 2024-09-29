2024.9.16

线性回归的features：
①tang=ppent/at
②at
③lev=(dlc+dltt)/at
④cashflow=(ib+dp)/at
⑤replace ceq=0 if missing(ceq)
replace txdb=0 if missing(txdb)
gen q=(at+csho*prcc_f-ceq-txdb)/at
如果上面这行结果不好就改成 q=(at-seq+csho*prcc_f)/at
⑥risk这是一个firm-year level的数，我等下放进我们的文件夹

这组跟RD应该是能特别好的predict的关系

然后把上面的ratio拆开再做一组 也就是
at
ppent
dlc
dltt
ib
dp
ceq (missing set to 0)
txdb (missing set to 0)
csho
prcc_f




以上三种训练输出分别命名为 approach0_*.csv approach1_*.csv approach2_*.csv