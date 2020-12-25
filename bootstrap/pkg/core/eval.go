/*
 Serene --- Yet an other Lisp

Copyright (c) 2020  Sameer Rahmani <lxsameer@gnu.org>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package core

import (
	"fmt"

	"serene-lang.org/bootstrap/pkg/ast"
)

func restOfExprs(es []IExpr, i int) []IExpr {
	if len(es)-1 > i {
		return es[i+1:]
	}
	return []IExpr{}
}

// evalForm evaluates the given expression `form` by a slightly different
// evaluation rules. For example if `form` is a list instead of the formal
// evaluation of a list it will evaluate all the elements and return the
// evaluated list
func evalForm(rt *Runtime, scope IScope, form IExpr) (IExpr, IError) {

	switch form.GetType() {
	case ast.Nil:
		return form, nil
	case ast.Number:
		return form, nil

	case ast.Fn:
		return form, nil

	case ast.String:
		return form, nil

	// Keyword evaluation rules:
	// * Keywords evaluates to themselves with respect to a
	// possible namespace alias. For example `::core/xyz`
	// will evaluates to `:serene.core/xyz` only if the ns
	// `serene.core` is loaded in the current ns with the
	// `core` alias. Also `::xyz` will evaluete to
	// `:<CURRENT_NS>/xyz`
	case ast.Keyword:
		// Eval initialize the keyword and MUTATES the state of the keyword
		// and returns the updated keyword which would be the same
		return form.(*Keyword).Eval(rt, scope)

	// Symbol evaluation rules:
	// * If it's a NS qualified symbol (NSQS), Look it up in the external symbol table of
	// the current namespace.
	// * If it's not a NSQS Look up the name in the current scope.
	// * Otherwise throw an error
	case ast.Symbol:
		var nsName string
		sym := form.(*Symbol)
		symbolName := sym.GetName()

		switch symbolName {
		case "true":
			return MakeTrue(MakeNodeFromExpr(form)), nil
		case "false":
			return MakeFalse(MakeNodeFromExpr(form)), nil
		case "nil":
			return MakeNil(MakeNodeFromExpr(form)), nil
		default:
			var expr *Binding
			if sym.IsNSQualified() {
				// Whether a namespace with the given alias loaded or not
				if !rt.CurrentNS().hasExternal(sym.GetNSPart()) {
					return nil, MakeErrorFor(rt, sym,
						fmt.Sprintf("Namespace '%s' is no loaded", sym.GetNSPart()),
					)
				}

				expr = rt.CurrentNS().LookupGlobal(rt, sym)
				nsName = sym.GetNSPart()
			} else {
				expr = scope.Lookup(rt, symbolName)
				nsName = rt.CurrentNS().GetName()
			}

			if expr == nil {
				return nil, MakeRuntimeErrorf(
					rt,
					"can't resolve symbol '%s' in ns '%s'",
					symbolName,
					nsName,
				)
			}

			return expr.Value, nil
		}

	// Evaluate all the elements in the list instead of following the lisp convention
	case ast.List:
		var result []IExpr

		lst := form.(*List)

		for {
			if lst.Count() > 0 {
				expr, err := EvalForms(rt, scope, lst.First())
				if err != nil {
					return nil, err
				}
				result = append(result, expr)
				lst = lst.Rest().(*List)
			} else {
				break
			}
		}
		return MakeList(result), nil
	}

	// Default case
	return nil, MakeError(rt, fmt.Sprintf("support for '%d' is not implemented", form.GetType()))
}

// EvalForms evaluates the given expr `expressions` (it can be a list, block, symbol or anything else)
// with the given runtime `rt` and the scope `scope`.
func EvalForms(rt *Runtime, scope IScope, expressions IExpr) (IExpr, IError) {
	// EvalForms is the main and the most important evaluation function on Serene.
	// It's a long loooooooooooong function. Why? Well, Because we don't want to
	// waste call stack spots in order to have a well organized code.
	// In order to avoid stackoverflows and implement TCO ( which is a must for
	// a functional language we need to avoid unnecessary calls and keep as much
	// as possible in a loop.
	//
	// `expressions` is argument is basically a tree of expressions which
	// this function walks over and rewrite it as necessary. The main purpose
	// of rewriting the tree is to eliminate any unnecessary function call.
	// This way we can eliminate tail calls and run everything faster.
	var ret IExpr
	var err IError

tco:
	for {
		// The TCO loop is there to take advantage or the fact that
		// in order to call a function or a block we simply can change
		// the value of the `expressions` and `scope`
		var exprs []IExpr

		// Block evaluation rules:
		// * If empty, return Nothing
		// * Otherwise evaluate the expressions in the block one by one
		//   and return the last result
		if expressions.GetType() == ast.Block {
			if expressions.(*Block).Count() == 0 {
				return &Nothing, nil
			}
			exprs = expressions.(*Block).ToSlice()
		} else {
			exprs = []IExpr{expressions}
		}

	body:
		for i := 0; i < len(exprs); i++ {
			//for i, forms := range exprs {
			forms := exprs[i]
			executionScope := forms.GetExecutionScope()
			scope := scope

			if executionScope != nil {
				scope = executionScope
			}

			if rt.IsDebugMode() {
				fmt.Printf("[DEBUG] Evaluating forms in NS: %s, Forms: %s\n", rt.CurrentNS().GetName(), forms)
				fmt.Printf("[DEBUG] -> State: I: %d, Exprs: %s\n", i, exprs)
			}

			// Evaluating forms one by one
			if forms.GetType() != ast.List {
				ret, err = evalForm(rt, scope, forms)
				continue body
			}

			// Expand macroes that exists in the given array of expression `forms`.
			// Since this implementation of Serene is an interpreter, the line
			// between compile time and runtime is unclear (afterall every thing
			// is happening in runtime). So we need to expand macroes before evaluating
			// other forms. In the future we might want to cache the expanded AST
			// as a cache and some sort of a bytecode for faster evaluation.
			forms, err = macroexpand(rt, scope, forms)
			if err != nil {
				return nil, err
			}

			if forms.GetType() != ast.List {
				return evalForm(rt, scope, forms)
			}

			list := forms.(*List)

			// Empty list evaluates to itself
			if list.Count() == 0 {
				ret = list
				break tco // return &Nil, nil
			}

			rawFirst := list.First()
			sform := ""

			// Handling special forms by looking up the first
			// element of the list. If it is a symbol, Grab
			// the name and check it for build it forms.
			//
			// Note: If we don't care about recursion in any
			// case we can simply extract it to a function
			// for example in `def` since we are going to
			// evaluate the value separately, we don't care
			// about recursion because we're going to handle
			// it wen we're evaluating the value. But in the
			// case of let it's a different story.
			if rawFirst.GetType() == ast.Symbol {
				sform = rawFirst.(*Symbol).GetName()
			}

			switch sform {

			// `ns` evaluation rules:
			// * The first element has to be a symbol representing the
			//   name of the namespace. ( We won't evaluate the first
			//   element )
			// TODO: decide on the syntax and complete the docs
			case "ns":
				ret, err = NSForm(rt, scope, list)
				continue body // no rewrite

			// `quote` evaluation rules:
			// * Only takes one argument
			// * Returns the argument without evaluating it
			case "quote":
				// Including the `quote` itself
				if list.Count() != 2 {
					return nil, MakeErrorFor(rt, list, "'quote' quote only accepts one argument.")
				}
				ret = list.Rest().First()
				err = nil
				continue body // no rewrite

			// case "quasiquote-expand":
			// 	return quasiquote(list.Rest().First()), nil

			// // For `quasiquote` evaluation rules, check out the documentation on
			// // the `quasiquote` function in `quasiquote.go`
			// case "quasiquote":
			// 	expressions = quasiquote(list.Rest().First())
			// 	continue tco // Loop over to execute the new expressions

			// TODO: Implement `list` in serene itself when we have destructuring available
			// Creates a new list form it's arguments.
			case "list":
				ret, err = evalForm(rt, scope, list.Rest().(*List))
				continue body // no rewrite

			// TODO: Implement `concat` in serene itself when we have protocols available
			// Concats all the collections together.
			case "concat":
				evaledForms, err := evalForm(rt, scope, list.Rest().(*List))

				if err != nil {
					return nil, err
				}

				lists := evaledForms.(*List).ToSlice()

				result := []IExpr{}
				for _, lst := range lists {
					if lst.GetType() != ast.List {
						return nil, MakeErrorFor(rt, lst, fmt.Sprintf("don't know how to concat '%s'", lst.String()))
					}

					result = append(result, lst.(*List).ToSlice()...)
				}
				ret, err = MakeList(result), nil
				continue body // no rewrite

			// TODO: Implement `list` in serene itself when we have destructuring available
			// Calls the `Cons` function on the second argument to cons the first arg to it.
			// In terms of a list, cons adds the first argument to as the new head of the list
			// given in the second argument.
			case "cons":
				if list.Count() != 3 {
					return nil, MakeErrorFor(rt, list, "'cons' needs exactly 3 arguments")
				}

				evaledForms, err := evalForm(rt, scope, list.Rest().(*List))

				if err != nil {
					return nil, err
				}
				coll, ok := evaledForms.(*List).Rest().First().(IColl)

				if !ok {
					return nil, MakeErrorFor(rt, list, "second arg of 'cons' has to be a collection")
				}

				ret, err = coll.Cons(evaledForms.(*List).First()), nil
				continue body // no rewrite

			// `def` evaluation rules
			// * The first argument has to be a symbol.
			// * The second argument has to be evaluated and be used as
			//   the value.
			// * Defines a global binding in the current namespace using
			//   the symbol name binded to the value
			case "def":
				ret, err = Def(rt, scope, list.Rest().(*List))
				continue body // no rewrite

			// `defmacro` evaluation rules:
			// * The first argument has to be a symbol
			// * The second argument has to be a list of argument for the macro
			// * The rest of the arguments will form a block that acts as the
			//   body of the macro.
			case "defmacro":
				ret, err = DefMacro(rt, scope, list.Rest().(*List))
				continue body // no rewrite

			// `macroexpand` evaluation rules:
			// * It has to have only one argument
			// * It WILL evaluate the only argument and tries to expand it
			//   as a macro and returns the expanded forms.
			case "macroexpand":
				if list.Count() != 2 {
					return nil, MakeErrorFor(rt, list, "'macroexpand' needs exactly one argument.")
				}
				evaledForm, e := evalForm(rt, scope, list.Rest().(*List))

				if e != nil {
					return nil, e
				}

				ret, err = macroexpand(rt, scope, evaledForm.(*List).First())
				continue body // no rewrite

			// `fn` evaluation rules:
			// * It needs at least a collection of arguments
			// * Defines an anonymous function.
			case "fn":
				ret, err = Fn(rt, scope, list.Rest().(*List))
				continue body // no rewrite

			// `if` evaluation rules:
			// * It has to get only 3 arguments: PRED THEN ELSE
			// * Evaluate only the PRED expression if the result
			//   is not `nil` or `false` evaluates THEN otherwise
			//   evaluate the ELSE expression and return the result.
			case "if":
				args := list.Rest().(*List)
				if args.Count() != 3 {
					return nil, MakeError(rt, "'if' needs exactly 3 aruments")
				}

				pred, err := EvalForms(rt, scope, args.First())
				result := pred.GetType()

				if err != nil {
					return nil, err
				}

				if result != ast.False && result != ast.Nil {
					// Truthy clause
					exprs = append([]IExpr{args.Rest().First()}, restOfExprs(exprs, i)...)
				} else {

					// Falsy clause
					exprs = append([]IExpr{args.Rest().Rest().First()}, restOfExprs(exprs, i)...)
				}
				i = 0
				goto body // rewrite

			// `do` evaluation rules:
			// * Evaluate the body as a new block in the TCO loop
			//   and return the result of the last expression
			case "do":
				// create a new slice of expressions by using the
				// do body and merging it by the remaining expressions
				// in the old `exprs` value and loop over it
				doExprs := list.Rest().(*List).ToSlice()
				exprs = append(doExprs, exprs[i+1:]...)
				i = 0
				goto body // rewrite

			// TODO: Implement `eval` as a native function
			// `eval` evaluation rules:
			// * It only takes on arguments.
			// * The argument has to be a form. For example if we pass a string
			//   to it as an argument that contains some expressions it will
			//   evaluate the string as string which will result to the same
			//   string. So IT DOES NOT READ the argument.
			// * It will evaluate the given form as the argument and return
			//   the result.
			case "eval":
				if list.Count() != 2 {
					return nil, MakeErrorFor(rt, list, "'eval' needs exactly 1 arguments")
				}
				form, err := evalForm(rt, scope, list.Rest().(*List))
				if err != nil {
					return nil, err
				}

				ret, err = EvalForms(rt, scope, form)
				continue body // no rewrite

			// `let` evaluation rules:
			// Let's assume the following:
			//   L = (let (A B C D) BODY)
			// * Create a new scope which has the current scope as the parent
			// * Evaluate the bindings by evaluating `B` and bind it to the name `A`
			//   in the scope.
			// * Repeat the prev step for expr D and name C
			// * Eval the block `BODY` using the created scope and return the result
			//   which is the result of the last expre in `BODY`
			case "let":
				if list.Count() < 2 {
					return nil, MakeError(rt, "'let' needs at list 1 aruments")
				}

				letScope := MakeScope(scope.(*Scope))

				// Since we're using IColl for the bindings, we can use either lists
				// or vectors or even hashmaps for bindings
				var bindings IColl
				bindings = list.Rest().First().(IColl)

				body := list.Rest().Rest().(*List).ToSlice()

				if bindings.Count()%2 != 0 {
					return nil, MakeError(rt, "'let' bindings has to have even number of forms.")
				}

				for {
					// We're reducing over bindings here
					if bindings.Count() == 0 {
						break
					}

					name := bindings.First()
					expr := bindings.Rest().First()

					// TODO: We need to destruct the bindings here and remove this check
					//       for the symbol type
					if name.GetType() != ast.Symbol {
						err := MakeErrorFor(rt, name, "'let' doesn't support desbbtructuring yet, use a symbol.")
						return nil, err
					}

					// You might be wondering why we're using `EvalForms` here to evaluate
					// the exprs in bindings, what about TCO ?
					// Well, It's called TAIL call optimization for a reason. Exprs in the
					// bindings are not tail calls
					evaluatedExpr, e := EvalForms(rt, letScope, expr)

					if e != nil {
						return nil, e
					}

					letScope.Insert(name.String(), evaluatedExpr, false)
					bindings = bindings.Rest().Rest().(IColl)
				}

				changeExecutionScope(body, letScope)
				exprs = append(body, exprs[i+1:]...)
				i = 0
				goto body

			// list evaluation rules:
			// * The first element of the list has to be an expression which is callable
			// * An empty list evaluates to itself.
			default:
				// Evaluating all the elements of the list
				listExprs, e := evalForm(rt, scope, list)
				if e != nil {
					err = e
					ret = nil
					break tco //return
				}

				f := listExprs.(*List).First()

				switch f.GetType() {
				case ast.Fn:
					// If the first element of the evaluated list is a function
					// create a scope for it by creating the binding to the given
					// parameters in the new scope and set the parent of it to
					// the scope which the function defined in and then set the
					// `expressions` to the body of function and loop again
					fn := f.(*Function)
					if e != nil {
						err = e
						ret = nil
						break body //return

					}

					argList := listExprs.(*List).Rest().(*List)

					fnScope, e := MakeFnScope(rt, fn.GetScope(), fn.GetParams(), argList)
					if e != nil {
						err = e
						ret = nil
						break body //return
					}

					body := fn.GetBody().ToSlice()
					changeExecutionScope(body, fnScope)
					exprs = append(body, restOfExprs(exprs, i)...)
					goto body // rewrite

				// If the function was a native function which is represented
				// by the `NativeFunction` struct
				case ast.NativeFn:
					fn := f.(*NativeFunction)
					ret, err = fn.Apply(
						rt,
						scope,
						MakeNodeFromExpr(fn),
						listExprs.(*List),
					)
					continue body // no rewrite

				default:
					err = MakeError(rt, "don't know how to execute anything beside function")
					ret = nil
					break tco
				}
			}
		}
		break tco
	}

	return ret, err
}

// Eval the given `Block` of code with the given runtime `rt`.
// The Important part here is that any expression that we need
// to Eval has to be wrapped in a Block. Don't confused the
// concept of Block with blocks from other languages which
// specify by using `{}` or indent or what ever. Blocks in terms
// of Serene are just arrays of expressions and nothing more.
func Eval(rt *Runtime, forms *Block) (IExpr, IError) {
	if forms.Count() == 0 {
		// Nothing is literally Nothing
		return &Nothing, nil
	}

	v, err := EvalForms(rt, rt.CurrentNS().GetRootScope(), forms)

	if err != nil {
		return nil, err
	}

	return v, nil
}

// EvalNSBody evals the body of the given namespace `ns` using the given
// runtime `rt`. It makes sure that the body starts with a `ns` special
// form with the same name as the ns argument.
func EvalNSBody(rt *Runtime, ns *Namespace) (*Namespace, IError) {
	body := ns.getForms()
	exprs := body.ToSlice()

	if len(exprs) == 0 {
		return nil, MakeError(rt, fmt.Sprintf("the 'ns' form is missing from '%s'", ns.GetName()))
	}

	if exprs[0].GetType() == ast.List {
		firstForm := exprs[0].(*List).First()
		if firstForm.GetType() == ast.Symbol && firstForm.(*Symbol).GetName() == "ns" {
			_, err := EvalForms(rt, ns.GetRootScope(), body)
			if err != nil {
				return nil, err
			}
			return ns, nil
		}
	}

	return nil, MakeError(rt, fmt.Sprintf("the 'ns' form is missing from '%s'", ns.GetName()))
}
