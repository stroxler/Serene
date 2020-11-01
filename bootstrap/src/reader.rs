/** Serene --- Yet an other Lisp
*
* Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 2 of the License.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
use crate::ast::Expr;
use crate::types::Number;
use std::io::{BufReader, Read};

pub type ReadResult<'a> = Result<Expr, String>;

pub struct ExprReader {
    location: i32,
    read_stack: Vec<char>,
}

impl ExprReader {
    fn new() -> ExprReader {
        ExprReader {
            location: 0,
            read_stack: vec![],
        }
    }

    /// Retun a single character by reading from the `reader`. ,
    ///
    /// # Arguments:
    ///
    /// * `reader`: The buffer to read from.
    /// * `skip_whitespace`: Whether or not to skip whitespace chars. *Bear in mind that
    ///   if you care about the newline char you should not skip the whitespaces*.
    fn get_char<T: Read>(
        &mut self,
        reader: &mut BufReader<T>,
        skip_whitespace: bool,
    ) -> Option<char> {
        loop {
            match self.read_stack.pop() {
                Some(c) if !c.is_whitespace() || !skip_whitespace => return Some(c),
                Some(_) => continue,
                None => (),
            };

            // Rust is weird, it doesn't provide a way to read from a buffer char by char.
            let mut single_char_buff = [0];
            let bytes_read = reader.read(&mut single_char_buff);
            match bytes_read {
                Ok(n) if n > 0 => self.location = self.location + 1,
                Ok(_) => return None,
                Err(_) => return None,
            };
            let ch = single_char_buff[0] as char;

            match ch {
                c if !c.is_whitespace() || !skip_whitespace => return Some(c),
                _ => (),
            };
        }
    }

    fn unget_char(&mut self, c: char) {
        self.read_stack.push(c);
    }

    // Look ahead. AFAIK Rust doesn't provide any unread functoinality like Java input streams which
    // sucks.
    fn peek_char<T: Read>(
        &mut self,
        reader: &mut BufReader<T>,
        skip_whitespace: bool,
    ) -> Option<char> {
        match self.get_char(reader, skip_whitespace) {
            Some(c) => {
                self.unget_char(c);
                Some(c)
            }
            None => None,
        }
    }

    fn read_quoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let rest = self.read_expr(reader)?;
        let elements = vec![Expr::make_symbol("quote".to_string()), rest];

        Ok(Expr::make_list(&elements))
    }

    fn read_unquoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.peek_char(reader, true) {
            Some('@') => {
                // Move forward in the buffer since we peeked it
                let _ = self.get_char(reader, true);
                let rest = self.read_expr(reader)?;
                let elements = vec![Expr::make_symbol("unquote-splicing".to_string()), rest];
                Ok(Expr::make_list(&elements))
            }
            _ => {
                let rest = self.read_expr(reader)?;
                let elements = vec![Expr::make_symbol("unquote".to_string()), rest];
                Ok(Expr::make_list(&elements))
            }
        }
    }

    fn read_quasiquoted_expr<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let rest = self.read_expr(reader)?;
        let elements = vec![Expr::make_symbol("quasiquote".to_string()), rest];
        Ok(Expr::make_list(&elements))
    }

    // TODO: We might want to replace Cons with an actual List struct
    fn read_list<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        let mut result = Expr::make_empty_list();

        match self.read_expr(reader) {
            Ok(value) => result.push(value),
            Err(e) => match self.get_char(reader, true) {
                Some(')') => return Ok(Expr::list_to_cons(result)),
                _ => return Err(e),
            },
        };

        loop {
            match self.get_char(reader, true) {
                Some(')') => return Ok(Expr::list_to_cons(result)),
                Some(e) => {
                    self.unget_char(e);
                    result.push(self.read_expr(reader)?)
                }
                None => return Err("Unexpected EOF while parsing a list.".to_string()),
            };
        }
    }

    fn is_valid_for_identifier(&self, c: char) -> bool {
        match c {
            '!'
            | '$'
            | '%'
            | '&'
            | '*'
            | '+'
            | '-'
            | '.'
            | '~'
            | '/'
            | ':'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '^'
            | '_'
            | 'a'..='z'
            | 'A'..='Z'
            | '0'..='9' => true,
            _ => false,
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
            }
            Some(e) => {
                return Err(format!(
                    "Unexpected character: got '{}', expected a symbol at {}",
                    e, self.location
                ))
            }
            None => return Err("Unexpected EOF".to_string()),
        };

        loop {
            match self.get_char(reader, false) {
                Some(v) if self.is_valid_for_identifier(v) => symbol.push(v),
                Some(v) => {
                    self.unget_char(v);
                    break;
                }
                None => break,
            }
        }

        Ok(Expr::make_symbol(symbol))
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
                _ => None,
            },
            None => None,
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
                        None => return Err(format!("Unexpected char to escape, got {}", e)),
                    },
                    //'\n' => return Err("Unescaped newlines are not allowed in string literals".to_string()),
                    _ => string.push(e),
                },
                None => return Err("Unexpected EOF while scanning a string".to_string()),
            }
        }
        Ok(Expr::make_string(string))
    }

    fn read_number<T: Read>(&mut self, reader: &mut BufReader<T>, neg: bool) -> ReadResult {
        let mut is_double = false;
        let mut string = (if neg { "-" } else { "" }).to_string();

        loop {
            match self.get_char(reader, false) {
                Some(e) if e == '.' && is_double => {
                    return Err("A double with more that one '.' ???".to_string())
                }
                Some(e) if e == '.' => {
                    is_double = true;
                    string.push(e);
                }
                Some(e) if e.is_digit(10) => string.push(e),
                Some(e) => {
                    self.unget_char(e);
                    break;
                }
                None => break,
            }
        }

        // TODO: Move this to ast module and use the `new` function on
        // Number struct
        if is_double {
            Ok(Expr::make_number(Number::Float(
                string.parse::<f64>().unwrap(),
            )))
        } else {
            Ok(Expr::make_number(Number::Integer(
                string.parse::<i64>().unwrap(),
            )))
        }
    }

    fn read_symbol<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.peek_char(reader, true) {
            Some(c) => match c {
                '\"' => self.read_string(reader),
                c if c.is_digit(10) => self.read_number(reader, false),

                // ':' => self.read_keyword(reader),
                '-' => {
                    // Read the '-' char
                    let _ = self.get_char(reader, true);
                    match self.peek_char(reader, true) {
                        Some(ch) => match ch {
                            ch if ch.is_digit(10) => self.read_number(reader, true),
                            _ => {
                                self.unget_char(c);
                                self._read_symbol(reader)
                            }
                        },
                        None => {
                            self.unget_char(c);
                            self._read_symbol(reader)
                        }
                    }
                }

                _ => self._read_symbol(reader),
            },
            None => Err("Unexpected EOF while scanning atom".to_string()),
        }
    }

    pub fn ignore_comments<T: Read>(&mut self, reader: &mut BufReader<T>) -> ReadResult {
        match self.get_char(reader, false) {
            Some(c) => match c {
                '\n' => Ok(Expr::Comment),
                _ => self.ignore_comments(reader),
            },
            None => Ok(Expr::Comment),
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
                    ';' => self.ignore_comments(reader),
                    //'[' => self.read_vector(reader),
                    //'{' => self.read_map(reader),
                    _ => {
                        self.unget_char(c);
                        self.read_symbol(reader)
                    }
                }
            }
            None => Ok(Expr::NoMatch),
        }
    }

    pub fn read_from_buffer<T: Read>(
        &mut self,
        reader: &mut BufReader<T>,
    ) -> Result<Vec<Expr>, String> {
        let mut ast = vec![];
        loop {
            match self.read_expr(reader) {
                Ok(Expr::NoMatch) => break,
                Err(v) => return Err(v),
                Ok(Expr::Comment) => continue,
                Ok(v) => ast.push(v),
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
