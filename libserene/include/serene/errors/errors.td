

class Error<string _desc = "", string _help = ""> {
      string desc = _desc;
      string help = _help;
}

// Examples of how to define a new error
// def Err : Error<"Err1 titel"> {
//     let description = [{
// err1
// multiline
// desc}];
// }

// def Err2 : Error {
//     let title = "err 2 titel";
//     let description = "err2 desc";
// }

// def Err3 : Error<"err3", [{
// err3
// multiline
// desc
// }]>;

def UnknownError: Error<"Can't find any description for this error.">;

def DefExpectSymbol: Error<"The first argument to 'def' has to be a Symbol.">;

def DefWrongNumberOfArgs: Error<"Wrong number of arguments is passed to the 'def' form.">;

def FnNoArgsList: Error<"'fn' form requires an argument list.">;

def FnArgsMustBeList: Error<"'fn' arguments should be a list.">;

def CantResolveSymbol: Error<"Can't resolve the given name.">;

def DontKnowHowToCallNode: Error<"Don't know how to call the given expression.">;

def PassFailureError: Error<"Pass Failure.">;

def NSLoadError: Error<"Faild to find a namespace.">;

def NSAddToSMError: Error<"Faild to add the namespace to the source manager.">;

def
    EOFWhileScaningAList: Error<"EOF reached before closing of list">;

def InvalidDigitForNumber: Error<"Invalid digit for a number.">;

def
    TwoFloatPoints: Error<"Two or more float point characters in a number">;

def
    InvalidCharacterForSymbol: Error<"Invalid character for a symbol">;

def CompilationError: Error<"Compilation error!">;

defvar errorsIndex = [
       UnknownError,
       DefExpectSymbol,
       DefWrongNumberOfArgs,
       FnNoArgsList,
       FnArgsMustBeList,
       CantResolveSymbol,
       DontKnowHowToCallNode,
       PassFailureError,
       NSLoadError,
       NSAddToSMError,
       EOFWhileScaningAList,
       InvalidDigitForNumber,
       TwoFloatPoints,
       InvalidCharacterForSymbol,
       CompilationError,
];
