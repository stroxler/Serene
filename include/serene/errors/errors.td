
class Error<string _title = "", string _desc = ""> {
      string title = _title;
      string description = _desc;
}

def Err : Error<"Err1 titel"> {
    let description = [{
err1
multiline
desc}];
}

def Err2 : Error {
    let title = "err 2 titel";
    let description = "err2 desc";
}

def Err3 : Error<"err3", [{
err3
multiline
desc
}]>;
