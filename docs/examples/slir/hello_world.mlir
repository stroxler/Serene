serene.ns @some.ns {
       // Value operator ----
       %0 = "serene.value"(){value= 3} : () -> i64
       // compact form
       %1 = serene.value 3 : i32

       %x = serene.symbol "some.ns" "x"
       // Def operator ----
       %foo = "serene.define"(%0){sym_name = "foo"}: (i64) -> !serene.symbol
       // compact form
       %bar = serene.define "bar", %0 : i64

       // Fn operator ----
       %f1 = "serene.fn"()({
       ^entry(%fnarg1 : i1, %fnarg2 : !serene.symbol):
              %2 = serene.value 3 : i32

              // Def operator ----
              %baz = "serene.define"(%fnarg1){sym_name = "baz"}: (i1) -> !serene.symbol
              serene.ret %baz : !serene.symbol
       },
       {
       ^b1(%f1 : i1):
              %3 = serene.value 4 : i32

              // Def operator ----
              %baz1 = "serene.define"(%3){sym_name = "baz"}: (i32) -> !serene.symbol
              serene.ret %baz1 : !serene.symbol
       ^b2:
              %baz2 = "serene.define"(%3){sym_name = "baz"}: (i32) -> !serene.symbol
              serene.ret %baz2 : !serene.symbol
       }){name = "some-fn", return_type = i32} : () -> !serene.fn

       %a1 = serene.value 1 : i1
       %a2 = serene.value "x" : !serene.symbol

       %result = serene.call %f1(%a1, %a2){} : (i1, !serene.symbol) -> i32
}
