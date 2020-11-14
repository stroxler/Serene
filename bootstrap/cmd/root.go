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
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"serene-lang.org/bootstrap/pkg/parser"
	"serene-lang.org/bootstrap/pkg/reader"
)

var cfgFile string

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "Serene",
	Short: "The bootstrap version",
	Long: `Serene's bootstrap interpreter V0.1.0

Serene's bootstrap interpreter is used to
bootstrap the Serene's compiler.'

It comes with ABSOLUTELY NO WARRANTY;
This is free software, and you are welcome
to redistribute it under certain conditions;
for details take a look at the LICENSE file.
`,
	Run: func(cmd *cobra.Command, args []string) {
		reader.ReadString("sameer mary")
		ast, _ := parser.ParseToAST("(asd 'mary '(1 2 3) `(asd ~asd ~@zxc))")
		fmt.Printf("%s\n", ast.String())
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize()
}
