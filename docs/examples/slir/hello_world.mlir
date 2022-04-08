module @some.ns {
       // Value operator ----
       %0 = "serene.value"(){value= 3} : () -> i64
       // compact form
       %1 = serene.value 3 : i32

       // Def operator ----
       %foo = "serene.def"(%0){name = "foo"}: (i64) -> !serene.symbol
       // compact form
       %bar = serene.def "bar", %0 : i64

       // Fn operator ----
       %f1 = "serene.fn"()({
       ^entry(%fnarg1 : i1, %fnarg2 : !serene.symbol, %fnarg3 : !serene.fn):
              %2 = serene.value 3 : i32

              // Def operator ----
              %baz = "serene.def"(%fnarg1){name = "baz"}: (i1) -> !serene.symbol
              serene.ret %baz : !serene.symbol
       },
       {
       ^b1(%f1 : i1):
              %3 = serene.value 4 : i32

              // Def operator ----
              %baz1 = "serene.def"(%3){name = "baz"}: (i32) -> !serene.symbol
              serene.ret %baz1 : !serene.symbol
       ^b2:
              %baz2 = "serene.def"(%3){name = "baz"}: (i32) -> !serene.symbol
              serene.ret %baz2 : !serene.symbol
       }){name = "some-fn", return_type = i32} : () -> !serene.fn
}
