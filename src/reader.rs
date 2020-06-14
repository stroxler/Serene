use crate::ast::{Expr, Number};
use std::io::{Read, BufReader};

pub type ReadResult = Result<Expr, String>;

pub struct ExprReader {
    read_stack: Vec<char>
}

impl ExprReader {

    fn new() -> ExprReader {
        ExprReader {
            read_stack: vec![]
        }
    }

    fn get_char<T: Read>(&mut self, reader: &mut BufReader<T>, skip_whitespace: bool) -> Option<char> {
        loop {
            match self.read_stack.pop() {
                Some(c) if !c.is_whitespace() || !skip_whitespace =>
                {
                    return Some(c)
                },
                Some(_) => continue,
                None => ()
            };

            // Rust is weird, it doesn't provide a way to read from a buffer char by char.
            let mut single_char_buff = [0];
            let bytes_read = reader.read(&mut single_char_buff);
            match bytes_read {
                Ok(n) if n > 0 => {},
                Ok(_) => return None,
                Err(_) => return None
            };
            let ch = single_char_buff[0] as char;

            match ch {
                c if !c.is_whitespace() || !skip_whitespace => return Some(c),
                _  => (),
            };
        }
    }

    fn unget_char(&mut self, c: char) {
        self.read_stack.push(c);
    }

    // Look ahead. AFAIK Rust doesn't provide any unread functoinality like Java input streams which
    // sucks.
    fn peek_char<T: Read>(&mut self, reader: &mut BufReader<T>, skip_whitespace: bool) -> Option<char> {
        match self.get_char(reader, skip_whitespace) {
            Some(c) => {
                self.unget_char(c);
                Some(c)
            },
            None => None
        }
    }

    fn read_quoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let rest = self.read_expr(reader)?;
        Ok(Expr::Cons(Box::new(Expr::Symbol("quote".to_string())), Box::new(rest)))
    }

    fn read_unquoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.peek_char(reader, true) {
            Some('@') => {
                // Move forward in the buffer since we peeked it
                let _ = self.get_char(reader, true);
                let rest = self.read_expr(reader)?;
                Ok(Expr::Cons(Box::new(Expr::Symbol("unquote-splicing".to_string())),
                              Box::new(rest)))
            },
            _ => {
                let rest = self.read_expr(reader)?;
                Ok(Expr::Cons(Box::new(Expr::Symbol("unquote".to_string())),
                              Box::new(rest)))
            }
        }
    }

    fn read_quasiquoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let rest = self.read_expr(reader)?;
        Ok(Expr::Cons(Box::new(Expr::Symbol("quasiquote".to_string())),
                      Box::new(rest)))
    }

    // TODO: We might want to replace Cons with an actual List struct
    fn read_list<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let first = match self.read_expr(reader) {
            Ok(value) => value,
            Err(e) => match self.get_char(reader, true) {
                // is it an empty list ?
                // TODO: we might want to return an actual empty list here
                Some(')') => return Ok(Expr::Nil),
                _ => return Err(e)
            }
        };
        let rest = match self.get_char(reader, true) {
            Some(e) => {
                self.unget_char(e);
                self.read_list(reader)?
            },
            None => return Err("Unexpected EOF, expected . or symbol".to_string())
        };

        Ok(Expr::Cons(Box::new(first), Box::new(rest)))
    }

    fn is_valid_for_identifier(&self, c: char) -> bool {
        match c {
            '!' | '$' | '%' | '&' | '*' | '+' | '-' | '.' | '~' |
            '/' | ':' | '<' | '=' | '>' | '?' | '@' | '^' | '_' |
            'a'..='z' | 'A'..='Z' | '0'..='9' => true,
            _ => false
        }
    }

    fn _read_symbol<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let mut symbol = match self.peek_char(reader, false) {
            Some(e) if self.is_valid_for_identifier(e) => {
                // Read into string
                let ch = self.get_char(reader, false).unwrap();
                let mut s = String::new();
                s.push(ch);
                s
            },
            Some(e) => {
                return Err(format!("Unexpected character: got {}, expected a symbol", e))
            },
            None => return Err("Unexpected EOF".to_string())
        };

        loop {
            match self.get_char(reader, false) {
                Some(v) if self.is_valid_for_identifier(v) => symbol.push(v),
                Some(v) => {
                    self.unget_char(v);
                    break;
                },
                None => break
            }
        }

        Ok(Expr::Symbol(symbol))
    }

    fn read_escape_char<T: Read>(&mut self, reader: &mut BufReader<T>) -> Option<char> {
        match self.get_char(reader, false) {
            Some(e) => match e {
                '\"' => Some('\"'),
                '\'' => Some('\''),
                '\\' => Some('\\'),
                'n' => Some('\n'),
                'r' => Some('\r'),
                't' => Some('\t'),
                _ => None
            },
            None => None
        }
    }

    fn read_string<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        // Skip the " char
        let _ = self.get_char(reader, false);

        let mut string = "".to_string();
        loop {
            match self.get_char(reader, false) {
                Some(e) => match e {
                    '\"' => break,
                    '\\' => match self.read_escape_char(reader) {
                        Some(v) => string.push(v),
                        None => return Err(format!("Unexpected char to escape, got {}", e))
                    },
                    //'\n' => return Err("Unescaped newlines are not allowed in string literals".to_string()),
                    _ => string.push(e)
                },
                None => return Err("Unexpected EOF while scanning a string".to_string())
            }
        }
        Ok(Expr::Str(string))
    }

    fn read_number<T: Read>(&mut self, reader: &mut BufReader<T>, neg: bool) -> ReadResult {
        let mut is_double = false;
        let mut string = "".to_string();
        loop {
            match self.get_char(reader, false) {
                Some(e) if e == '.' && is_double => return Err("A double with more that one '.' ???".to_string()),
                Some(e) if e == '.' => {
                    is_double = true;
                    string.push(e);
                }
                Some(e) if e.is_digit(10) => string.push(e),
                Some(e) => {
                    self.unget_char(e);
                    break;
                }
                None => break
            }
        }
        if is_double {
            Ok(Expr::Num(Number::Float(string.parse::<f64>().unwrap())))
        } else {
            Ok(Expr::Num(Number::Integer(string.parse::<i64>().unwrap())))
        }
    }

    fn read_symbol<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.peek_char(reader, true) {
            Some(c) => match c {
                '\"' => self.read_string(reader),
                c if c.is_digit(10) => self.read_number(reader, false),

                // ':' => self.read_keyword(reader),
                // '-' => {
                // Neg number
                // }
                _ => self._read_symbol(reader)
            },
            None => Err("Unexpected EOF while scanning atom".to_string())
        }
    }

    pub fn read_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.get_char(reader, true) {
            Some(c) => {
                match c {
                    '\'' => self.read_quoted_expr(reader),
                    '~' => self.read_unquoted_expr(reader),
                    '`' => self.read_quasiquoted_expr(reader),
                    '(' => self.read_list(reader),
                    //'[' => self.read_vector(reader),
                    //'{' => self.read_map(reader),
                    _  => {
                        self.unget_char(c);
                        self.read_symbol(reader)
                    }
                }
            },
            None => Ok(Expr::NoMatch)
        }
    }

    pub fn read_from_buffer<T: Read>(&mut self, reader: &mut BufReader<T>) -> Result<Vec<Expr>, String> {
        let mut ast = vec![];
        loop {
            match self.read_expr(reader) {
                Ok(Expr::NoMatch) => break,
                Err(v) => return Err(v),
                Ok(v) => ast.push(v)
            }
        }
        Ok(ast)
    }

    pub fn read(&mut self, string: &str) -> Result<Vec<Expr>, String> {
        let reader = BufReader::new(string.as_bytes());
        let mut buf_reader = BufReader::new(reader);
        self.read_from_buffer(&mut buf_reader)
    }

}


pub fn read_string(input: &str) -> Result<Vec<Expr>, String> {
    let mut reader = ExprReader::new();
    reader.read(input)
}
