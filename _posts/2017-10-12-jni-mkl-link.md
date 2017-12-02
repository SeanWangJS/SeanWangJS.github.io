---
layout: default
---

## Java 本地调用连接 Intel 数学核心库(MKL)

```
javac .\jblas\src\main\java\com\haswalk\jblas\JBlas.java -h .\jblas\src\main\java\com\haswalk\jblas\lib

gcc -c .\jblas\src\main\java\com\haswalk\jblas\lib\com_haswalk_jblas_JBlas.c -o .\jblas\src\main\java\com\haswalk\jblas\lib\com_haswalk_jblas_JBlas.o -I "%MKL_ROOT%\include" -I "%JAVA_HOME%\include" -I "%JAVA_HOME%\include\win32"

gcc -shared -o .\jblas\src\main\java\com\haswalk\jblas\lib\com_haswalk_jblas_JBlas.dll .\jblas\src\main\java\com\haswalk\jblas\lib\com_haswalk_jblas_JBlas.o -L "%MKL_ROOT%\lib\intel64_win" -lmkl_rt
```